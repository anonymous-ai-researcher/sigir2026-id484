"""
Dataset Module for OntoEL

Handles data loading, preprocessing, and batching for biomedical entity linking.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple, Any
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MentionExample:
    """Represents a single mention with its context and gold entity."""
    mention_id: str
    mention_text: str
    context_left: str
    context_right: str
    gold_entity_id: str
    gold_entity_name: str
    gold_type_ids: List[str] = field(default_factory=list)
    candidates: Optional[List[str]] = None


@dataclass
class EntityInfo:
    """Information about an entity."""
    entity_id: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    type_ids: List[str] = field(default_factory=list)
    description: Optional[str] = None


class BioELDataset(Dataset):
    """PyTorch Dataset for Biomedical Entity Linking."""
    
    def __init__(
        self,
        examples: List[MentionExample],
        entities: Dict[str, EntityInfo],
        tokenizer: PreTrainedTokenizer,
        type_to_idx: Dict[str, int],
        entity_to_idx: Dict[str, int],
        max_seq_length: int = 128,
        context_window: int = 64,
        num_hard_negatives: int = 15,
        include_synonyms: bool = True,
        mode: str = "train",
    ):
        self.examples = examples
        self.entities = entities
        self.tokenizer = tokenizer
        self.type_to_idx = type_to_idx
        self.entity_to_idx = entity_to_idx
        self.idx_to_entity = {v: k for k, v in entity_to_idx.items()}
        self.max_seq_length = max_seq_length
        self.context_window = context_window
        self.num_hard_negatives = num_hard_negatives
        self.include_synonyms = include_synonyms
        self.mode = mode
        self.num_types = len(type_to_idx)
        self.num_entities = len(entity_to_idx)
        
        self.entity_type_memberships = self._compute_entity_type_memberships()
        self.all_entity_ids = list(entities.keys())
    
    def _compute_entity_type_memberships(self) -> torch.Tensor:
        """Pre-compute binary type membership matrix for all entities."""
        memberships = torch.zeros(self.num_entities, self.num_types)
        
        for entity_id, info in self.entities.items():
            if entity_id in self.entity_to_idx:
                entity_idx = self.entity_to_idx[entity_id]
                for type_id in info.type_ids:
                    if type_id in self.type_to_idx:
                        type_idx = self.type_to_idx[type_id]
                        memberships[entity_idx, type_idx] = 1.0
        
        return memberships
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def _tokenize_mention(self, example: MentionExample) -> Dict[str, torch.Tensor]:
        """Tokenize mention with context."""
        mention_with_context = (
            f"{example.context_left} {example.mention_text} {example.context_right}"
        )
        
        encoded = self.tokenizer(
            mention_with_context,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
    
    def _get_type_labels(self, entity_id: str) -> torch.Tensor:
        """Get type labels for an entity from TBox."""
        info = self.entities.get(entity_id)
        labels = torch.zeros(self.num_types)
        
        if info is not None:
            for type_id in info.type_ids:
                if type_id in self.type_to_idx:
                    labels[self.type_to_idx[type_id]] = 1.0
        
        return labels
    
    def _sample_negatives(self, example: MentionExample) -> List[str]:
        """Sample negative candidates."""
        gold_id = example.gold_entity_id
        negatives = []
        
        # Use pre-retrieved hard negatives if available
        if example.candidates:
            hard_negs = [c for c in example.candidates if c != gold_id]
            negatives.extend(hard_negs[:self.num_hard_negatives])
        
        # Fill remaining with random negatives
        num_random = max(0, self.num_hard_negatives - len(negatives))
        if num_random > 0:
            available = [e for e in self.all_entity_ids if e != gold_id and e not in negatives]
            random_negs = random.sample(available, min(num_random, len(available)))
            negatives.extend(random_negs)
        
        return negatives
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training/evaluation example."""
        example = self.examples[idx]
        
        mention_encoded = self._tokenize_mention(example)
        
        gold_id = example.gold_entity_id
        gold_idx = self.entity_to_idx.get(gold_id, 0)
        
        type_labels = self._get_type_labels(gold_id)
        
        item = {
            "mention_input_ids": mention_encoded["input_ids"],
            "mention_attention_mask": mention_encoded["attention_mask"],
            "gold_entity_idx": gold_idx,
            "type_labels": type_labels,
            "mention_id": example.mention_id,
        }
        
        if self.mode == "train":
            negatives = self._sample_negatives(example)
            candidate_ids = [gold_id] + negatives
            candidate_indices = [self.entity_to_idx.get(c, 0) for c in candidate_ids]
            
            positive_mask = torch.zeros(len(candidate_ids))
            positive_mask[0] = 1.0
            
            item.update({
                "candidate_indices": torch.tensor(candidate_indices),
                "positive_mask": positive_mask,
                "num_candidates": len(candidate_ids),
            })
        
        elif self.mode in ["eval", "test"]:
            if example.candidates:
                candidate_ids = example.candidates
                candidate_indices = [self.entity_to_idx.get(c, 0) for c in candidate_ids]
                
                positive_mask = torch.zeros(len(candidate_ids))
                for i, c in enumerate(candidate_ids):
                    if c == gold_id:
                        positive_mask[i] = 1.0
                
                item.update({
                    "candidate_indices": torch.tensor(candidate_indices),
                    "positive_mask": positive_mask,
                    "num_candidates": len(candidate_ids),
                })
        
        return item


class BioELCollator:
    """Custom collator for BioEL batches."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        entity_type_memberships: torch.Tensor,
        max_candidates: int = 64,
    ):
        self.tokenizer = tokenizer
        self.entity_type_memberships = entity_type_memberships
        self.max_candidates = max_candidates
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples."""
        batch_size = len(batch)
        
        mention_input_ids = torch.stack([b["mention_input_ids"] for b in batch])
        mention_attention_mask = torch.stack([b["mention_attention_mask"] for b in batch])
        type_labels = torch.stack([b["type_labels"] for b in batch])
        
        result = {
            "mention_input_ids": mention_input_ids,
            "mention_attention_mask": mention_attention_mask,
            "type_labels": type_labels,
        }
        
        if "candidate_indices" in batch[0]:
            max_cands = min(
                max(b["num_candidates"] for b in batch),
                self.max_candidates
            )
            
            candidate_indices = torch.zeros(batch_size, max_cands, dtype=torch.long)
            positive_mask = torch.zeros(batch_size, max_cands)
            
            for i, b in enumerate(batch):
                num_cands = min(b["num_candidates"], max_cands)
                candidate_indices[i, :num_cands] = b["candidate_indices"][:num_cands]
                positive_mask[i, :num_cands] = b["positive_mask"][:num_cands]
            
            # Get type memberships for candidates
            candidate_type_memberships = self.entity_type_memberships[candidate_indices]
            
            result.update({
                "candidate_indices": candidate_indices,
                "positive_mask": positive_mask,
                "candidate_type_memberships": candidate_type_memberships,
            })
        
        return result


def load_examples_from_jsonl(filepath: str) -> List[MentionExample]:
    """Load mention examples from JSONL file."""
    examples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            example = MentionExample(
                mention_id=data.get("mention_id", str(len(examples))),
                mention_text=data["mention_text"],
                context_left=data.get("context_left", ""),
                context_right=data.get("context_right", ""),
                gold_entity_id=data["gold_entity_id"],
                gold_entity_name=data.get("gold_entity_name", ""),
                gold_type_ids=data.get("gold_type_ids", []),
                candidates=data.get("candidates"),
            )
            examples.append(example)
    return examples


def load_entities_from_jsonl(filepath: str) -> Dict[str, EntityInfo]:
    """Load entity information from JSONL file."""
    entities = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            entity = EntityInfo(
                entity_id=data["entity_id"],
                name=data["name"],
                synonyms=data.get("synonyms", []),
                type_ids=data.get("type_ids", []),
                description=data.get("description"),
            )
            entities[entity.entity_id] = entity
    return entities


def load_type_hierarchy(filepath: str) -> Tuple[Dict[str, int], torch.Tensor]:
    """
    Load type hierarchy and disjointness from JSON file.
    
    Returns:
        type_to_idx: Mapping from type ID to index
        disjointness_matrix: Binary matrix of disjoint type pairs
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    type_names = data["types"]
    type_to_idx = {name: i for i, name in enumerate(type_names)}
    
    num_types = len(type_names)
    disjointness_matrix = torch.zeros(num_types, num_types)
    
    for pair in data.get("disjoint_pairs", []):
        i = type_to_idx.get(pair[0])
        j = type_to_idx.get(pair[1])
        if i is not None and j is not None:
            disjointness_matrix[i, j] = 1
            disjointness_matrix[j, i] = 1
    
    return type_to_idx, disjointness_matrix


def create_dataloaders(
    train_file: str,
    dev_file: str,
    entities_file: str,
    type_hierarchy_file: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 64,
    max_seq_length: int = 128,
    num_hard_negatives: int = 15,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train and dev dataloaders.
    
    Returns:
        train_loader, dev_loader, metadata dict
    """
    # Load data
    train_examples = load_examples_from_jsonl(train_file)
    dev_examples = load_examples_from_jsonl(dev_file)
    entities = load_entities_from_jsonl(entities_file)
    type_to_idx, disjointness_matrix = load_type_hierarchy(type_hierarchy_file)
    
    # Create entity index mapping
    entity_to_idx = {eid: i for i, eid in enumerate(entities.keys())}
    
    # Create datasets
    train_dataset = BioELDataset(
        examples=train_examples,
        entities=entities,
        tokenizer=tokenizer,
        type_to_idx=type_to_idx,
        entity_to_idx=entity_to_idx,
        max_seq_length=max_seq_length,
        num_hard_negatives=num_hard_negatives,
        mode="train",
    )
    
    dev_dataset = BioELDataset(
        examples=dev_examples,
        entities=entities,
        tokenizer=tokenizer,
        type_to_idx=type_to_idx,
        entity_to_idx=entity_to_idx,
        max_seq_length=max_seq_length,
        mode="eval",
    )
    
    # Create collator
    collator = BioELCollator(
        tokenizer=tokenizer,
        entity_type_memberships=train_dataset.entity_type_memberships,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    metadata = {
        "entities": entities,
        "entity_to_idx": entity_to_idx,
        "type_to_idx": type_to_idx,
        "disjointness_matrix": disjointness_matrix,
        "entity_type_memberships": train_dataset.entity_type_memberships,
        "num_entities": len(entities),
        "num_types": len(type_to_idx),
    }
    
    return train_loader, dev_loader, metadata
