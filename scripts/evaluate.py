#!/usr/bin/env python3
"""
Evaluation Script for OntoEL

Usage:
    python scripts/evaluate.py --config configs/sample_config.yaml --checkpoint checkpoints/best_model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import json
import logging
from tqdm import tqdm

from src.model import OntoEL
from src.dataset import (
    BioELDataset,
    BioELCollator,
    load_examples_from_jsonl,
    load_entities_from_jsonl,
    load_type_hierarchy,
)
from src.trainer import compute_detailed_metrics
from src.utils import load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OntoEL model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions",
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    setup_logging()
    set_seed(config.get("seed", 42))
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    encoder_name = config.get("model", {}).get(
        "encoder_name",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    # Load data
    paths = config.get("paths", {})
    if args.split == "test":
        eval_file = paths.get("test_file", "data/sample/test.jsonl")
    else:
        eval_file = paths.get("dev_file", "data/sample/dev.jsonl")
    
    entities_file = paths.get("entities_file", "data/sample/entities.jsonl")
    type_hierarchy_file = paths.get("type_hierarchy_file", "data/sample/type_hierarchy.json")
    
    logger.info(f"Loading data from {eval_file}")
    eval_examples = load_examples_from_jsonl(eval_file)
    entities = load_entities_from_jsonl(entities_file)
    type_to_idx, disjointness_matrix = load_type_hierarchy(type_hierarchy_file)
    
    entity_to_idx = {eid: i for i, eid in enumerate(entities.keys())}
    
    logger.info(f"Evaluation examples: {len(eval_examples)}")
    
    # Create dataset
    data_config = config.get("data", {})
    eval_dataset = BioELDataset(
        examples=eval_examples,
        entities=entities,
        tokenizer=tokenizer,
        type_to_idx=type_to_idx,
        entity_to_idx=entity_to_idx,
        max_seq_length=data_config.get("max_seq_length", 128),
        mode="eval",
    )
    
    collator = BioELCollator(
        tokenizer=tokenizer,
        entity_type_memberships=eval_dataset.entity_type_memberships,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.get("training", {}).get("batch_size", 64),
        shuffle=False,
        collate_fn=collator,
        num_workers=config.get("num_workers", 0),
    )
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model_config = config.get("model", {})
    fuzzy_config = config.get("fuzzy_logic", {})
    fusion_config = config.get("fusion", {})
    
    model = OntoEL(
        encoder_name=encoder_name,
        hidden_dim=model_config.get("hidden_dim", 768),
        projection_dim=model_config.get("projection_dim", 768),
        num_types=len(type_to_idx),
        sigmoid_sharpness=fuzzy_config.get("sigmoid_sharpness", 10.0),
        fusion_alpha=fusion_config.get("alpha", 0.8),
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / "model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Set buffers
    model.set_entity_type_memberships(eval_dataset.entity_type_memberships)
    model.set_disjointness_matrix(disjointness_matrix)
    
    # Compute type embeddings
    type_names = list(type_to_idx.keys())
    model.precompute_type_embeddings(type_names)
    
    # Evaluate
    logger.info("Running evaluation...")
    metrics = compute_detailed_metrics(model, eval_loader, device)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy@1: {metrics['accuracy@1']:.4f}")
    print(f"Accuracy@5: {metrics['accuracy@5']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"Median Rank: {metrics['median_rank']:.2f}")
    print("-" * 50)
    print(f"Neural-only Accuracy: {metrics['neural_only_accuracy']:.4f}")
    print(f"Onto-only Accuracy: {metrics['onto_only_accuracy']:.4f}")
    print(f"Improvement from Onto: {metrics['improvement_from_onto']:.4f}")
    print("-" * 50)
    print(f"Onto helped: {metrics['onto_helped_count']} cases")
    print(f"Onto hurt: {metrics['onto_hurt_count']} cases")
    print(f"Net benefit: {metrics['onto_net_benefit']} cases")
    print("=" * 50)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return metrics


if __name__ == "__main__":
    main()
