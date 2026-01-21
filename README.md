# OntoEL: Neuro-Symbolic Biomedical Entity Linking with Differentiable Fuzzy EL++ Reasoning

A PyTorch implementation of OntoEL, a neuro-symbolic framework for biomedical entity linking that integrates differentiable fuzzy Description Logic reasoning with neural retrieval.

## Overview

OntoEL addresses the challenge of biomedical entity linking by combining:
- **Neural Bi-Encoder Retrieval**: Uses SapBERT for initial candidate generation based on semantic similarity
- **Fuzzy EL++ Reasoning**: Applies differentiable logical constraints for ontology-aware re-ranking
- **Context-Aware Type Inference**: Infers semantic types from mention context using projected embeddings

The key innovation is treating ontological axioms as differentiable soft constraints, enabling end-to-end training while resolving the "implication bias" problem in fuzzy logic.

## Installation

### Requirements

```bash
# Python 3.8+
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
scikit-learn>=1.2.0
pyyaml>=6.0
```

## Quick Start

### 1. Prepare Sample Data

The repository includes sample data for testing. For full experiments, download the datasets:

```bash
# Run preprocessing with sample data
python scripts/preprocess_data.py --dataset sample --output_dir data/processed

# Or download and preprocess MedMentions
python scripts/download_datasets.py --dataset medmentions
python scripts/preprocess_data.py --dataset medmentions --output_dir data/processed
```

### 2. Train the Model

```bash
# Train with sample data (quick test)
python scripts/train.py --config configs/sample_config.yaml

# Train on MedMentions
python scripts/train.py --config configs/medmentions_config.yaml
```

### 3. Run Inference

```bash
# Evaluate on test set
python scripts/evaluate.py --config configs/sample_config.yaml --checkpoint checkpoints/best_model.pt

# Interactive demo
python scripts/demo.py --checkpoint checkpoints/best_model.pt
```

## Project Structure

```
ontoel/
├── README.md
├── requirements.txt
├── configs/
│   ├── default_config.yaml      # Default hyperparameters
│   ├── sample_config.yaml       # Config for sample data
│   └── medmentions_config.yaml  # Config for MedMentions
├── data/
│   ├── sample/                  # Sample data for testing
│   │   ├── mentions.jsonl
│   │   ├── entities.jsonl
│   │   └── type_hierarchy.json
│   └── processed/               # Preprocessed data
├── src/
│   ├── __init__.py
│   ├── model.py                 # OntoEL model architecture
│   ├── fuzzy_logic.py           # Fuzzy EL++ operators
│   ├── type_inference.py        # Context-aware type inference
│   ├── dataset.py               # Data loading and processing
│   ├── retrieval.py             # Candidate retrieval with FAISS
│   ├── trainer.py               # Training loop
│   └── utils.py                 # Utility functions
├── scripts/
│   ├── preprocess_data.py       # Data preprocessing
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── demo.py                  # Interactive demo
└── checkpoints/                 # Saved models
```

## Model Architecture

### 1. Neural Bi-Encoder (Retrieval Stage)

```
Mention + Context → SapBERT Encoder → Mention Embedding (768-dim)
Entity Name → SapBERT Encoder → Entity Embedding (768-dim)

Neural Score = cosine_similarity(mention_emb, entity_emb)
```

### 2. Fuzzy Logic Layer (Re-ranking Stage)

```
Context → Type Inference Module → Fuzzy Type Memberships τ^I(m)
Candidate → TBox Lookup → Crisp Type Memberships τ^I(e)

Consistency Score = Π_τ I_σ(τ^I(m), τ^I(e))
```

Where `I_σ` is the Sigmoidal Reichenbach implication:
```
I_σ(a, b) = σ(s · (1 - a + ab - 0.5))
```

### 3. Score Fusion

```
Final Score = α · normalized_neural_score + (1-α) · onto_score
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder_name` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Pretrained encoder |
| `hidden_dim` | 768 | Encoder hidden dimension |
| `projection_dim` | 768 | Type projection dimension |
| `sigmoid_sharpness` | 10 | Sharpness (s) for Sigmoidal Reichenbach |
| `fusion_alpha` | 0.8 | Balance between neural and logic scores |
| `margin` | 0.2 | Margin for ranking loss |
| `type_loss_weight` | 0.5 | Weight (λ) for type prediction loss |
| `learning_rate` | 2e-5 | AdamW learning rate |
| `batch_size` | 64 | Training batch size |
| `epochs` | 10 | Number of training epochs |
| `top_k` | 64 | Number of candidates to retrieve |
| `hard_negatives` | 15 | Hard negatives per sample |

## Key Components

### Fuzzy Logic Operators

**Product T-Norm (Conjunction)**:
```python
def product_tnorm(a, b):
    return a * b
```

**Probabilistic Sum (Existential)**:
```python
def prob_sum(values):
    return 1 - torch.prod(1 - values, dim=-1)
```

**Sigmoidal Reichenbach Implication**:
```python
def sigmoidal_reichenbach(a, b, s=10):
    linear = 1 - a + a * b
    return torch.sigmoid(s * (linear - 0.5))
```

### Type Inference

The model infers semantic types from context using dual projection:
```python
m_proj = W_m @ mention_embedding
t_proj = W_t @ type_name_embedding
membership = sigmoid(dot(m_proj, t_proj) / temperature)
```

## Training

The training objective combines ranking loss and type prediction loss:

```
L = L_rank + λ · L_type
```

Where:
- `L_rank`: Margin-based ranking loss encouraging gold entity over negatives
- `L_type`: Binary cross-entropy for type prediction

## Evaluation Metrics

- **Accuracy@1**: Percentage of mentions with correct top-1 prediction
- **Accuracy@5**: Percentage of mentions with correct entity in top-5
- **MRR**: Mean Reciprocal Rank
- **Recall@k**: Percentage of mentions where gold entity is in top-k candidates

## Citation

```bibtex
@inproceedings{ontoel2026,
  title={OntoEL: Neuro-Symbolic Biomedical Entity Linking with Differentiable Fuzzy EL++ Reasoning},
  author={Anonymous},
  booktitle={Proceedings of SIGIR},
  year={2026}
}
```

## License

MIT License
