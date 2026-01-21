"""
Utility Functions for OntoEL

Common utilities for configuration, logging, reproducibility, and evaluation.
"""

import torch
import numpy as np
import random
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import os


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Supports inheritance via _base_ key.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Handle inheritance
    if "_base_" in config:
        base_path = config_path.parent / config["_base_"]
        base_config = load_config(str(base_path))
        
        # Merge configs (child overrides parent)
        config = deep_merge(base_config, config)
        del config["_base_"]
    
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_file: Specific log file name
    """
    handlers = [logging.StreamHandler()]
    
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            from datetime import datetime
            log_file = f"ontoel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        handlers.append(logging.FileHandler(log_dir / log_file))
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


@dataclass
class TrainingArguments:
    """Training arguments dataclass."""
    # Model
    encoder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    hidden_dim: int = 768
    projection_dim: int = 768
    num_types: int = 21
    
    # Fuzzy logic
    sigmoid_sharpness: float = 10.0
    fusion_alpha: float = 0.8
    
    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 64
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Loss
    margin: float = 0.2
    type_loss_weight: float = 0.5
    num_hard_negatives: int = 15
    
    # Retrieval
    top_k: int = 64
    
    # Data
    max_seq_length: int = 128
    
    # Paths
    data_dir: str = "data/processed"
    output_dir: str = "checkpoints"
    
    # Hardware
    device: str = "cuda"
    fp16: bool = False
    bf16: bool = False
    num_workers: int = 4
    
    # Misc
    seed: int = 42
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    early_stopping: bool = True
    patience: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingArguments":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TrainingArguments":
        """Create from nested config dict."""
        flat_config = {}
        
        for section, values in config.items():
            if isinstance(values, dict):
                flat_config.update(values)
            else:
                flat_config[section] = values
        
        return cls.from_dict(flat_config)


def get_device(device: str = "auto") -> torch.device:
    """
    Get the appropriate device.
    
    Args:
        device: Device string ("auto", "cuda", "cpu", "mps")
        
    Returns:
        torch.device
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics dictionary as string."""
    parts = []
    for key, value in metrics.items():
        if prefix:
            key = f"{prefix}_{key}"
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
        else:
            parts.append(f"{key}: {value}")
    return " | ".join(parts)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change to qualify as improvement
            mode: "max" for metrics to maximize, "min" for minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def save_predictions(
    predictions: List[Dict[str, Any]],
    output_path: str,
):
    """
    Save model predictions to file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Output file path
    """
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def load_predictions(input_path: str) -> List[Dict[str, Any]]:
    """Load predictions from file."""
    predictions = []
    with open(input_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


class MetricTracker:
    """Track metrics over training."""
    
    def __init__(self, metrics: List[str]):
        """
        Initialize metric tracker.
        
        Args:
            metrics: List of metric names to track
        """
        self.metrics = metrics
        self.history = {m: [] for m in metrics}
        self.best = {m: 0.0 for m in metrics}
    
    def update(self, values: Dict[str, float]):
        """Update with new metric values."""
        for metric in self.metrics:
            if metric in values:
                self.history[metric].append(values[metric])
                if values[metric] > self.best[metric]:
                    self.best[metric] = values[metric]
    
    def get_best(self) -> Dict[str, float]:
        """Get best values for all metrics."""
        return self.best.copy()
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest values for all metrics."""
        return {m: self.history[m][-1] if self.history[m] else 0.0 for m in self.metrics}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "history": self.history,
            "best": self.best,
        }


def compute_recall_at_k(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    k: int,
) -> float:
    """
    Compute Recall@k.
    
    Args:
        scores: Prediction scores [batch, num_candidates]
        positive_mask: Binary mask for positives [batch, num_candidates]
        k: Number of top predictions to consider
        
    Returns:
        Recall@k value
    """
    topk_indices = scores.topk(min(k, scores.shape[1]), dim=-1).indices
    
    # Check if any positive is in top-k
    batch_size = scores.shape[0]
    recall = 0.0
    
    for i in range(batch_size):
        positive_indices = positive_mask[i].nonzero(as_tuple=True)[0]
        topk = topk_indices[i]
        
        if any(idx in topk for idx in positive_indices):
            recall += 1.0
    
    return recall / batch_size
