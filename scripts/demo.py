#!/usr/bin/env python3
"""
Interactive Demo for OntoEL

Demonstrates the neuro-symbolic entity linking process with visualization
of neural vs. ontological scores.

Usage:
    python scripts/demo.py --checkpoint checkpoints/best_model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoTokenizer
import json

from src.model import OntoEL
from src.dataset import load_entities_from_jsonl, load_type_hierarchy


def run_demo(
    model: OntoEL,
    tokenizer,
    entities: dict,
    entity_to_idx: dict,
    type_to_idx: dict,
    device: str,
):
    """Run interactive demo."""
    idx_to_entity = {v: k for k, v in entity_to_idx.items()}
    
    print("\n" + "=" * 60)
    print("OntoEL: Neuro-Symbolic Biomedical Entity Linking Demo")
    print("=" * 60)
    print("\nThis demo shows how OntoEL combines neural similarity")
    print("with fuzzy ontological reasoning for entity linking.")
    print("\nType 'quit' to exit.\n")
    
    while True:
        # Get input
        print("-" * 60)
        mention = input("Enter mention text: ").strip()
        if mention.lower() == "quit":
            break
        
        context_left = input("Enter left context: ").strip()
        context_right = input("Enter right context: ").strip()
        
        # Tokenize
        full_text = f"{context_left} {mention} {context_right}"
        encoded = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        # Get all candidate IDs (simplified: use all entities)
        all_entity_ids = list(entities.keys())[:20]  # Limit for demo
        candidate_indices = torch.tensor(
            [[entity_to_idx[eid] for eid in all_entity_ids]]
        ).to(device)
        
        # Get type memberships
        candidate_types = torch.zeros(1, len(all_entity_ids), len(type_to_idx)).to(device)
        for i, eid in enumerate(all_entity_ids):
            for tid in entities[eid].type_ids:
                if tid in type_to_idx:
                    candidate_types[0, i, type_to_idx[tid]] = 1.0
        
        # Run model
        with torch.no_grad():
            outputs = model(
                mention_input_ids=input_ids,
                mention_attention_mask=attention_mask,
                candidate_ids=candidate_indices,
                candidate_type_memberships=candidate_types,
                return_components=True,
            )
        
        # Display results
        scores = outputs["scores"][0].cpu()
        neural = outputs["normalized_neural"][0].cpu()
        onto = outputs["onto_scores"][0].cpu()
        type_probs = outputs["type_probs"][0].cpu()
        
        print("\n" + "=" * 60)
        print("INFERRED TYPE PROBABILITIES FROM CONTEXT")
        print("=" * 60)
        for tid, prob in zip(type_to_idx.keys(), type_probs):
            bar = "█" * int(prob * 20)
            print(f"{tid:20s} {prob:.3f} |{bar}")
        
        print("\n" + "=" * 60)
        print("TOP-5 CANDIDATES")
        print("=" * 60)
        print(f"{'Rank':<5} {'Entity':<25} {'Neural':<10} {'Onto':<10} {'Final':<10} {'Types'}")
        print("-" * 80)
        
        # Sort by final score
        sorted_indices = scores.argsort(descending=True)
        for rank, idx in enumerate(sorted_indices[:5]):
            idx = idx.item()
            eid = all_entity_ids[idx]
            entity = entities[eid]
            types_str = ", ".join(entity.type_ids)
            
            print(
                f"{rank+1:<5} {entity.name[:24]:<25} "
                f"{neural[idx]:.4f}     {onto[idx]:.4f}     {scores[idx]:.4f}     {types_str}"
            )
        
        print("\n")


def main():
    parser = argparse.ArgumentParser(description="OntoEL Interactive Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sample_config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()
    
    # Load config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.utils import load_config
    config = load_config(args.config)
    
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    encoder_name = config.get("model", {}).get(
        "encoder_name",
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    )
    print(f"Loading tokenizer: {encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    
    # Load data
    paths = config.get("paths", {})
    entities_file = paths.get("entities_file", "data/sample/entities.jsonl")
    type_hierarchy_file = paths.get("type_hierarchy_file", "data/sample/type_hierarchy.json")
    
    entities = load_entities_from_jsonl(entities_file)
    type_to_idx, disjointness_matrix = load_type_hierarchy(type_hierarchy_file)
    entity_to_idx = {eid: i for i, eid in enumerate(entities.keys())}
    
    print(f"Loaded {len(entities)} entities, {len(type_to_idx)} types")
    
    # Create model
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
    
    # Load checkpoint if specified
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint) / "model.pt"
        if checkpoint_path.exists():
            print(f"Loading checkpoint: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            print("Running with untrained model (for demo purposes)")
    else:
        print("No checkpoint specified, running with untrained model")
    
    model = model.to(device)
    model.eval()
    
    # Set up type embeddings
    type_names = list(type_to_idx.keys())
    model.precompute_type_embeddings(type_names)
    model.set_disjointness_matrix(disjointness_matrix)
    
    # Create entity type memberships
    entity_type_memberships = torch.zeros(len(entities), len(type_to_idx))
    for eid, entity in entities.items():
        idx = entity_to_idx[eid]
        for tid in entity.type_ids:
            if tid in type_to_idx:
                entity_type_memberships[idx, type_to_idx[tid]] = 1.0
    model.set_entity_type_memberships(entity_type_memberships)
    
    # Run demo
    run_demo(model, tokenizer, entities, entity_to_idx, type_to_idx, device)


if __name__ == "__main__":
    main()
