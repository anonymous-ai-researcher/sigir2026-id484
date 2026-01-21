"""
Trainer Module for OntoEL

Implements the complete training pipeline with:
- Multi-task learning (ranking + type prediction)
- Hard negative mining
- Gradient accumulation and mixed precision
- Evaluation and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Tuple, Any, Callable
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime

from .model import OntoEL, OntoELForTraining


logger = logging.getLogger(__name__)


class OntoELTrainer:
    """
    Trainer for OntoEL model.
    
    Handles:
    - Training loop with gradient accumulation
    - Mixed precision training (FP16/BF16)
    - Learning rate scheduling with warmup
    - Evaluation and metric computation
    - Checkpointing and early stopping
    """
    
    def __init__(
        self,
        model: OntoELForTraining,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        # Optimization
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
        # Scheduler
        scheduler_type: str = "linear",
        warmup_ratio: float = 0.1,
        # Training
        num_epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        # Mixed precision
        fp16: bool = False,
        bf16: bool = False,
        # Logging
        logging_steps: int = 100,
        eval_steps: int = 500,
        save_steps: int = 1000,
        # Output
        output_dir: str = "checkpoints",
        # Early stopping
        early_stopping: bool = True,
        patience: int = 3,
        # Device
        device: str = "cuda",
    ):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        
        # Optimization settings
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Training settings
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        
        # Output
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.early_stopping = early_stopping
        self.patience = patience
        self.best_metric = 0.0
        self.patience_counter = 0
        
        # Mixed precision
        self.fp16 = fp16
        self.bf16 = bf16
        self.use_amp = fp16 or bf16
        self.scaler = GradScaler() if fp16 else None
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(adam_beta1, adam_beta2, adam_epsilon)
        
        # Calculate total steps
        self.total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        self.warmup_steps = int(warmup_ratio * self.total_steps)
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler_type)
        
        # Tracking
        self.global_step = 0
        self.current_epoch = 0
        self.training_history = []
    
    def _create_optimizer(
        self,
        adam_beta1: float,
        adam_beta2: float,
        adam_epsilon: float,
    ) -> AdamW:
        """Create optimizer with weight decay fix."""
        # Separate parameters that should not have weight decay
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
        )
    
    def _create_scheduler(self, scheduler_type: str):
        """Create learning rate scheduler."""
        if scheduler_type == "linear":
            # Linear decay with warmup
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.total_steps - self.warmup_steps,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.warmup_steps],
            )
        
        elif scheduler_type == "cosine":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
        
        else:
            # Constant learning rate
            return None
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Training history and final metrics
        """
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Warmup steps: {self.warmup_steps}")
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            epoch_loss = self._train_epoch()
            
            # Evaluate
            metrics = self.evaluate()
            
            logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Loss: {epoch_loss:.4f} - "
                f"Acc@1: {metrics['accuracy@1']:.4f} - "
                f"MRR: {metrics['mrr']:.4f}"
            )
            
            # Save checkpoint
            self.save_checkpoint(f"epoch_{epoch + 1}")
            
            # Early stopping check
            if self.early_stopping:
                if metrics["accuracy@1"] > self.best_metric:
                    self.best_metric = metrics["accuracy@1"]
                    self.patience_counter = 0
                    self.save_checkpoint("best_model")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
        
        return {
            "training_history": self.training_history,
            "best_metric": self.best_metric,
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self._training_step(batch)
            total_loss += loss
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss:.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}" if self.scheduler else f"{self.learning_rate:.2e}",
            })
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                self.training_history.append({
                    "step": self.global_step,
                    "loss": loss,
                    "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.learning_rate,
                })
            
            # Evaluation during training
            if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                metrics = self.evaluate()
                logger.info(
                    f"Step {self.global_step} - Acc@1: {metrics['accuracy@1']:.4f}"
                )
                self.model.train()
            
            # Save checkpoint
            if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                self.save_checkpoint(f"step_{self.global_step}")
        
        return total_loss / num_batches
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Forward pass with optional mixed precision
        if self.use_amp:
            with autocast(dtype=self.amp_dtype):
                outputs = self.model.training_step(
                    mention_input_ids=batch["mention_input_ids"],
                    mention_attention_mask=batch["mention_attention_mask"],
                    candidate_ids=batch["candidate_indices"],
                    positive_mask=batch["positive_mask"],
                    type_labels=batch["type_labels"],
                    candidate_type_memberships=batch.get("candidate_type_memberships"),
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps
        else:
            outputs = self.model.training_step(
                mention_input_ids=batch["mention_input_ids"],
                mention_attention_mask=batch["mention_attention_mask"],
                candidate_ids=batch["candidate_indices"],
                positive_mask=batch["positive_mask"],
                type_labels=batch["type_labels"],
                candidate_type_memberships=batch.get("candidate_type_memberships"),
            )
            loss = outputs["loss"] / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
        
        self.global_step += 1
        
        return loss.item() * self.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on dev set.
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        all_scores = []
        all_positive_masks = []
        all_type_probs = []
        all_type_labels = []
        
        for batch in tqdm(self.dev_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = self.model(
                mention_input_ids=batch["mention_input_ids"],
                mention_attention_mask=batch["mention_attention_mask"],
                candidate_ids=batch["candidate_indices"],
                candidate_type_memberships=batch.get("candidate_type_memberships"),
                return_components=True,
            )
            
            all_scores.append(outputs["scores"].cpu())
            all_positive_masks.append(batch["positive_mask"].cpu())
            all_type_probs.append(outputs["type_probs"].cpu())
            all_type_labels.append(batch["type_labels"].cpu())
        
        # Concatenate all batches
        all_scores = torch.cat(all_scores, dim=0)
        all_positive_masks = torch.cat(all_positive_masks, dim=0)
        all_type_probs = torch.cat(all_type_probs, dim=0)
        all_type_labels = torch.cat(all_type_labels, dim=0)
        
        # Compute metrics
        metrics = self._compute_metrics(
            all_scores, all_positive_masks, all_type_probs, all_type_labels
        )
        
        return metrics
    
    def _compute_metrics(
        self,
        scores: torch.Tensor,
        positive_mask: torch.Tensor,
        type_probs: torch.Tensor,
        type_labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Get predictions (argmax of scores)
        predictions = scores.argmax(dim=-1)
        
        # Get gold positions (argmax of positive_mask)
        gold_positions = positive_mask.argmax(dim=-1)
        
        # Accuracy@1
        correct_at_1 = (predictions == gold_positions).float().mean().item()
        
        # Accuracy@5
        top5_indices = scores.topk(min(5, scores.shape[1]), dim=-1).indices
        correct_at_5 = (
            (top5_indices == gold_positions.unsqueeze(-1)).any(dim=-1).float().mean().item()
        )
        
        # MRR (Mean Reciprocal Rank)
        sorted_indices = scores.argsort(dim=-1, descending=True)
        ranks = (sorted_indices == gold_positions.unsqueeze(-1)).nonzero(as_tuple=True)[1] + 1
        mrr = (1.0 / ranks.float()).mean().item()
        
        # Type prediction accuracy
        type_preds = (type_probs > 0.5).float()
        type_accuracy = (type_preds == type_labels).float().mean().item()
        
        return {
            "accuracy@1": correct_at_1,
            "accuracy@5": correct_at_5,
            "mrr": mrr,
            "type_accuracy": type_accuracy,
        }
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt",
        )
        
        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_metric": self.best_metric,
            },
            checkpoint_dir / "training_state.pt",
        )
        
        # Save training history
        with open(checkpoint_dir / "history.json", "w") as f:
            json.dump(self.training_history, f)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model state
        self.model.load_state_dict(
            torch.load(checkpoint_dir / "model.pt", map_location=self.device)
        )
        
        # Load optimizer state
        state = torch.load(checkpoint_dir / "training_state.pt", map_location=self.device)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler and state["scheduler"]:
            self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.current_epoch = state["current_epoch"]
        self.best_metric = state["best_metric"]
        
        # Load training history
        if (checkpoint_dir / "history.json").exists():
            with open(checkpoint_dir / "history.json") as f:
                self.training_history = json.load(f)
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")


def compute_detailed_metrics(
    model: OntoEL,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Compute detailed evaluation metrics including per-type analysis.
    
    Args:
        model: Trained OntoEL model
        dataloader: Evaluation dataloader
        device: Device to use
        
    Returns:
        Dictionary with detailed metrics
    """
    model.eval()
    
    results = {
        "correct_at_1": [],
        "correct_at_5": [],
        "ranks": [],
        "neural_correct": [],
        "onto_correct": [],
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing detailed metrics"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            outputs = model(
                mention_input_ids=batch["mention_input_ids"],
                mention_attention_mask=batch["mention_attention_mask"],
                candidate_ids=batch["candidate_indices"],
                candidate_type_memberships=batch.get("candidate_type_memberships"),
                return_components=True,
            )
            
            scores = outputs["scores"]
            neural_scores = outputs["normalized_neural"]
            onto_scores = outputs["onto_scores"]
            positive_mask = batch["positive_mask"]
            
            # Get gold position
            gold_pos = positive_mask.argmax(dim=-1)
            
            # Final predictions
            predictions = scores.argmax(dim=-1)
            correct = (predictions == gold_pos).cpu().tolist()
            results["correct_at_1"].extend(correct)
            
            # Top-5 accuracy
            top5 = scores.topk(min(5, scores.shape[1]), dim=-1).indices
            correct_5 = (top5 == gold_pos.unsqueeze(-1)).any(dim=-1).cpu().tolist()
            results["correct_at_5"].extend(correct_5)
            
            # Ranks
            sorted_idx = scores.argsort(dim=-1, descending=True)
            for i in range(len(gold_pos)):
                rank = (sorted_idx[i] == gold_pos[i]).nonzero(as_tuple=True)[0].item() + 1
                results["ranks"].append(rank)
            
            # Neural-only predictions
            neural_preds = neural_scores.argmax(dim=-1)
            neural_correct = (neural_preds == gold_pos).cpu().tolist()
            results["neural_correct"].extend(neural_correct)
            
            # Onto-only predictions
            onto_preds = onto_scores.argmax(dim=-1)
            onto_correct = (onto_preds == gold_pos).cpu().tolist()
            results["onto_correct"].extend(onto_correct)
    
    # Compute summary statistics
    summary = {
        "accuracy@1": np.mean(results["correct_at_1"]),
        "accuracy@5": np.mean(results["correct_at_5"]),
        "mrr": np.mean([1.0 / r for r in results["ranks"]]),
        "mean_rank": np.mean(results["ranks"]),
        "median_rank": np.median(results["ranks"]),
        "neural_only_accuracy": np.mean(results["neural_correct"]),
        "onto_only_accuracy": np.mean(results["onto_correct"]),
        "improvement_from_onto": np.mean(results["correct_at_1"]) - np.mean(results["neural_correct"]),
    }
    
    # Analyze where onto helped
    onto_helped = sum(
        1 for c, nc in zip(results["correct_at_1"], results["neural_correct"])
        if c and not nc
    )
    onto_hurt = sum(
        1 for c, nc in zip(results["correct_at_1"], results["neural_correct"])
        if not c and nc
    )
    
    summary["onto_helped_count"] = onto_helped
    summary["onto_hurt_count"] = onto_hurt
    summary["onto_net_benefit"] = onto_helped - onto_hurt
    
    return summary
