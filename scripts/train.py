#!/usr/bin/env python3
"""
Training Script for OntoEL

Usage:
    python scripts/train.py --config configs/sample_config.yaml
    python scripts/train.py --config configs/medmentions_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging

from src.model import OntoELForTraining
from src.dataset import (
    BioELDataset,
    BioELCollator,
    load_examples_from_jsonl,
    load_entities_from_jsonl,
    load_type_hierarchy,
)
from src.trainer import OntoELTrainer
from src.utils import load_config, set_seed, setup_logging, count_parameters

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train OntoEL model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sample_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    setup_logging(config.get("paths", {}).get("log_dir", "logs"))
    set_seed(config.get("seed", 42))
    
    # Get device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    encoder_name = config.get("model", {}).get(
        "encoder_name",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    logger.info(f"Loading tokenizer: {encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    # Load data paths
    paths = config.get("paths", {})
    train_file = paths.get("train_file", "data/sample/train.jsonl")
    dev_file = paths.get("dev_file", "data/sample/dev.jsonl")
    entities_file = paths.get("entities_file", "data/sample/entities.jsonl")
    type_hierarchy_file = paths.get("type_hierarchy_file", "data/sample/type_hierarchy.json")
    
    # Load data
    logger.info("Loading data...")
    train_examples = load_examples_from_jsonl(train_file)
    dev_examples = load_examples_from_jsonl(dev_file)
    entities = load_entities_from_jsonl(entities_file)
    type_to_idx, disjointness_matrix = load_type_hierarchy(type_hierarchy_file)
    
    logger.info(f"Train examples: {len(train_examples)}")
    logger.info(f"Dev examples: {len(dev_examples)}")
    logger.info(f"Entities: {len(entities)}")
    logger.info(f"Types: {len(type_to_idx)}")
    
    # Create entity index mapping
    entity_to_idx = {eid: i for i, eid in enumerate(entities.keys())}
    
    # Get config values
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    loss_config = config.get("loss", {})
    model_config = config.get("model", {})
    fuzzy_config = config.get("fuzzy_logic", {})
    fusion_config = config.get("fusion", {})
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = BioELDataset(
        examples=train_examples,
        entities=entities,
        tokenizer=tokenizer,
        type_to_idx=type_to_idx,
        entity_to_idx=entity_to_idx,
        max_seq_length=data_config.get("max_seq_length", 128),
        num_hard_negatives=loss_config.get("num_hard_negatives", 15),
        mode="train",
    )
    
    dev_dataset = BioELDataset(
        examples=dev_examples,
        entities=entities,
        tokenizer=tokenizer,
        type_to_idx=type_to_idx,
        entity_to_idx=entity_to_idx,
        max_seq_length=data_config.get("max_seq_length", 128),
        mode="eval",
    )
    
    # Create collator
    collator = BioELCollator(
        tokenizer=tokenizer,
        entity_type_memberships=train_dataset.entity_type_memberships,
    )
    
    # Create dataloaders
    batch_size = training_config.get("batch_size", 64)
    num_workers = config.get("num_workers", 0)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
    )
    
    # Create model
    logger.info("Creating model...")
    model = OntoELForTraining(
        encoder_name=encoder_name,
        hidden_dim=model_config.get("hidden_dim", 768),
        projection_dim=model_config.get("projection_dim", 768),
        num_types=len(type_to_idx),
        sigmoid_sharpness=fuzzy_config.get("sigmoid_sharpness", 10.0),
        fusion_alpha=fusion_config.get("alpha", 0.8),
        temperature_init=model_config.get("temperature_init", 3.32),
        margin=loss_config.get("margin", 0.2),
        type_loss_weight=loss_config.get("type_loss_weight", 0.5),
    )
    
    # Set entity embeddings and type memberships
    model.set_entity_type_memberships(train_dataset.entity_type_memberships)
    model.set_disjointness_matrix(disjointness_matrix)
    
    # Compute type name embeddings
    type_names = list(type_to_idx.keys())
    logger.info(f"Computing embeddings for {len(type_names)} type names...")
    model = model.to(device)
    model.precompute_type_embeddings(type_names)
    
    # Count parameters
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts}")
    
    # Create trainer
    output_dir = paths.get("output_dir", "checkpoints")
    
    trainer = OntoELTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        learning_rate=training_config.get("learning_rate", 2e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        scheduler_type=training_config.get("scheduler", "linear"),
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        num_epochs=training_config.get("num_epochs", 10),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", False),
        logging_steps=training_config.get("logging_steps", 100),
        eval_steps=training_config.get("eval_steps", 500),
        save_steps=training_config.get("save_steps", 1000),
        output_dir=output_dir,
        early_stopping=training_config.get("early_stopping", True),
        patience=training_config.get("patience", 3),
        device=device,
    )
    
    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    logger.info("Starting training...")
    result = trainer.train()
    
    logger.info(f"Training complete! Best metric: {result['best_metric']:.4f}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    logger.info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
