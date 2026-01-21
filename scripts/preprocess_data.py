#!/usr/bin/env python3
"""
Data Preprocessing Script for OntoEL

Preprocesses various biomedical entity linking datasets into the OntoEL format.

Usage:
    python scripts/preprocess_data.py --dataset medmentions --output_dir data/processed
    python scripts/preprocess_data.py --dataset bc5cdr --output_dir data/processed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# UMLS Semantic Types for MedMentions ST21pv
ST21PV_SEMANTIC_TYPES = {
    "T005": "Virus",
    "T007": "Bacterium",
    "T017": "Anatomical Structure",
    "T022": "Body System",
    "T031": "Body Substance",
    "T033": "Finding",
    "T037": "Injury or Poisoning",
    "T038": "Biologic Function",
    "T058": "Health Care Activity",
    "T062": "Research Activity",
    "T074": "Medical Device",
    "T082": "Spatial Concept",
    "T091": "Biomedical Occupation or Discipline",
    "T092": "Organization",
    "T097": "Professional or Occupational Group",
    "T098": "Population Group",
    "T103": "Chemical",
    "T168": "Food",
    "T170": "Intellectual Product",
    "T201": "Clinical Attribute",
    "T204": "Eukaryote",
}

# Semantic Group mappings
SEMANTIC_GROUPS = {
    "Anatomy": ["T017", "T022", "T031"],
    "Chemicals_Drugs": ["T103"],
    "Disorders": ["T037"],
    "Living_Beings": ["T005", "T007", "T204"],
    "Objects": ["T074"],
    "Occupations": ["T091", "T097"],
    "Organizations": ["T092"],
    "Phenomena": ["T038"],
    "Physiology": ["T201"],
    "Procedures": ["T058", "T062"],
    "Concepts": ["T082", "T170", "T168", "T033"],
}


def preprocess_medmentions(
    input_dir: str,
    output_dir: str,
    use_semantic_groups: bool = False,
):
    """
    Preprocess MedMentions dataset.
    
    Expected input structure:
    - input_dir/corpus_pubtator.txt (PubTator format)
    - input_dir/st21pv/ (ST21pv annotations)
    
    Args:
        input_dir: Directory containing MedMentions files
        output_dir: Output directory for processed files
        use_semantic_groups: Whether to use coarse semantic groups
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing MedMentions from {input_dir}")
    
    # For now, create a placeholder that explains the expected format
    readme_content = """
# MedMentions Preprocessing

To use this script with MedMentions:

1. Download MedMentions from: https://github.com/chanzuckerberg/MedMentions

2. Place files in input_dir:
   - corpus_pubtator.txt
   - st21pv/train.txt
   - st21pv/dev.txt
   - st21pv/test.txt

3. Run: python scripts/preprocess_data.py --dataset medmentions --input_dir /path/to/medmentions

The script will parse PubTator format and convert to JSONL format for OntoEL.
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create type hierarchy
    if use_semantic_groups:
        types = list(SEMANTIC_GROUPS.keys())
    else:
        types = list(ST21PV_SEMANTIC_TYPES.values())
    
    type_hierarchy = {
        "types": types,
        "type_names": {t: t for t in types},
        "disjoint_pairs": [],
        "subsumption": [],
    }
    
    # Add disjointness between all type pairs (simplification)
    for i, t1 in enumerate(types):
        for t2 in types[i+1:]:
            type_hierarchy["disjoint_pairs"].append([t1, t2])
    
    with open(output_dir / "type_hierarchy.json", "w") as f:
        json.dump(type_hierarchy, f, indent=2)
    
    logger.info(f"Created type hierarchy with {len(types)} types")
    logger.info(f"Output written to {output_dir}")


def create_sample_data(output_dir: str):
    """
    Create sample data for testing the pipeline.
    
    This generates synthetic biomedical entity linking examples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating sample data in {output_dir}")
    
    # Sample entities with realistic biomedical concepts
    entities = [
        {
            "entity_id": "C0009443",
            "name": "Common cold",
            "synonyms": ["Cold", "Acute coryza", "Head cold", "Nasopharyngitis"],
            "type_ids": ["Disorder"],
        },
        {
            "entity_id": "C0234192",
            "name": "Cold sensation",
            "synonyms": ["Feeling cold", "Cold feeling", "Sensation of cold"],
            "type_ids": ["Finding"],
        },
        {
            "entity_id": "C0032285",
            "name": "Pneumonia",
            "synonyms": ["Lung inflammation", "Pulmonary infection"],
            "type_ids": ["Disorder"],
        },
        {
            "entity_id": "C0015967",
            "name": "Fever",
            "synonyms": ["Pyrexia", "Elevated temperature", "Febrile"],
            "type_ids": ["Finding"],
        },
        {
            "entity_id": "C0004057",
            "name": "Aspirin",
            "synonyms": ["Acetylsalicylic acid", "ASA"],
            "type_ids": ["Chemical"],
        },
        {
            "entity_id": "C0024109",
            "name": "Lung",
            "synonyms": ["Pulmo", "Pulmonary structure"],
            "type_ids": ["Anatomy"],
        },
        {
            "entity_id": "C0543467",
            "name": "Surgery",
            "synonyms": ["Surgical procedure", "Operation"],
            "type_ids": ["Procedure"],
        },
    ]
    
    # Sample mentions demonstrating ambiguity
    train_examples = [
        {
            "mention_id": "train_0",
            "mention_text": "cold",
            "context_left": "The patient reported feeling",
            "context_right": "after the procedure",
            "gold_entity_id": "C0234192",
            "gold_entity_name": "Cold sensation",
            "gold_type_ids": ["Finding"],
            "candidates": ["C0234192", "C0009443", "C0015967"],
        },
        {
            "mention_id": "train_1",
            "mention_text": "cold",
            "context_left": "She was diagnosed with a",
            "context_right": "and prescribed rest",
            "gold_entity_id": "C0009443",
            "gold_entity_name": "Common cold",
            "gold_type_ids": ["Disorder"],
            "candidates": ["C0009443", "C0234192", "C0032285"],
        },
        {
            "mention_id": "train_2",
            "mention_text": "pneumonia",
            "context_left": "X-ray showed",
            "context_right": "in the right lower lobe",
            "gold_entity_id": "C0032285",
            "gold_entity_name": "Pneumonia",
            "gold_type_ids": ["Disorder"],
            "candidates": ["C0032285", "C0024109", "C0009443"],
        },
        {
            "mention_id": "train_3",
            "mention_text": "fever",
            "context_left": "The child presented with high",
            "context_right": "and cough",
            "gold_entity_id": "C0015967",
            "gold_entity_name": "Fever",
            "gold_type_ids": ["Finding"],
            "candidates": ["C0015967", "C0234192", "C0032285"],
        },
        {
            "mention_id": "train_4",
            "mention_text": "aspirin",
            "context_left": "Treatment included",
            "context_right": "for pain relief",
            "gold_entity_id": "C0004057",
            "gold_entity_name": "Aspirin",
            "gold_type_ids": ["Chemical"],
            "candidates": ["C0004057", "C0543467", "C0024109"],
        },
    ]
    
    dev_examples = train_examples[:2]  # Use subset for dev
    test_examples = train_examples[2:4]  # Use subset for test
    
    # Write entities
    with open(output_dir / "entities.jsonl", "w") as f:
        for entity in entities:
            f.write(json.dumps(entity) + "\n")
    
    # Write examples
    for split, examples in [("train", train_examples), ("dev", dev_examples), ("test", test_examples)]:
        with open(output_dir / f"{split}.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
    
    # Create type hierarchy
    types = ["Finding", "Disorder", "Chemical", "Anatomy", "Procedure"]
    type_hierarchy = {
        "types": types,
        "type_names": {t: t for t in types},
        "disjoint_pairs": [
            ["Finding", "Disorder"],
            ["Finding", "Chemical"],
            ["Finding", "Anatomy"],
            ["Finding", "Procedure"],
            ["Disorder", "Chemical"],
            ["Disorder", "Anatomy"],
            ["Disorder", "Procedure"],
            ["Chemical", "Anatomy"],
            ["Chemical", "Procedure"],
            ["Anatomy", "Procedure"],
        ],
        "subsumption": [],
    }
    
    with open(output_dir / "type_hierarchy.json", "w") as f:
        json.dump(type_hierarchy, f, indent=2)
    
    logger.info(f"Created {len(entities)} entities")
    logger.info(f"Created {len(train_examples)} train, {len(dev_examples)} dev, {len(test_examples)} test examples")


def main():
    parser = argparse.ArgumentParser(description="Preprocess data for OntoEL")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["medmentions", "bc5cdr", "ncbi", "sample"],
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory containing raw data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--use_semantic_groups",
        action="store_true",
        help="Use coarse semantic groups instead of fine-grained types",
    )
    args = parser.parse_args()
    
    if args.dataset == "sample":
        create_sample_data(args.output_dir)
    elif args.dataset == "medmentions":
        if args.input_dir is None:
            logger.warning("No input_dir specified, creating placeholder files")
        preprocess_medmentions(
            args.input_dir or "data/raw/medmentions",
            args.output_dir,
            args.use_semantic_groups,
        )
    else:
        logger.error(f"Preprocessing for {args.dataset} not yet implemented")
        logger.info("Use --dataset sample to create sample data for testing")


if __name__ == "__main__":
    main()
