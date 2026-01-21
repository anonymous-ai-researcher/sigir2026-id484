<p align="center">
  <img src="assets/ontoel_logo.svg" alt="OntoEL Logo" width="400"/>
</p>

<h1 align="center">🧬 OntoEL</h1>

<p align="center">
  <strong>Neuro-Symbolic Biomedical Entity Linking with Differentiable Fuzzy EL++ Reasoning</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/🤗_Transformers-4.30+-yellow?style=for-the-badge" alt="Transformers"></a>
  <a href="https://github.com/facebookresearch/faiss"><img src="https://img.shields.io/badge/FAISS-1.7+-blue?style=for-the-badge" alt="FAISS"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome">
  <img src="https://img.shields.io/badge/Maintained-Yes-blue.svg?style=flat-square" alt="Maintained">
</p>

---

## 🎯 Overview

**OntoEL** is a state-of-the-art neuro-symbolic framework that bridges the gap between neural retrieval and logical reasoning for biomedical entity linking. Unlike traditional approaches that treat ontologies as flat dictionaries, OntoEL leverages the rich terminological knowledge (TBox) encoded in biomedical ontologies through differentiable fuzzy Description Logic.

<p align="center">
  <img src="assets/architecture_overview.png" alt="OntoEL Architecture" width="800"/>
</p>

### ✨ Key Features

| Feature                      | Description                                                      |
| ---------------------------- | ---------------------------------------------------------------- |
| 🧠 **Neuro-Symbolic Fusion** | Combines neural bi-encoder retrieval with fuzzy EL++ reasoning   |
| 🎯 **Context-Aware Typing**  | Infers semantic types from mention context using dual projection |
| ⚡ **Gradient-Stable Logic**  | Sigmoidal Reichenbach implication resolves "implication bias"    |
| 🔄 **End-to-End Training**   | Ontological axioms as differentiable soft constraints            |
| 🚀 **Efficient Inference**   | 30x faster than cross-encoders with comparable accuracy          |

### 🏆 Performance Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOTA Results on BioEL Benchmarks             │
├─────────────────────────────────────────────────────────────────┤
│  📊 MedMentions ST21pv    │  Acc@1: 87.8%  │  +4.2% vs baseline │
│  💊 BC5CDR                │  Acc@1: 90.5%  │  +2.5% vs baseline │
│  🏥 NCBI-Disease          │  Acc@1: 89.8%  │  +1.9% vs baseline │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

Ensure you have Python 3.8+ installed. We recommend using a virtual environment:

```bash
# Create and activate virtual environment
python -m venv ontoel-env
source ontoel-env/bin/activate  # On Windows: ontoel-env\Scripts\activate

# Clone the repository
git clone https://github.com/anonymous/ontoel.git
cd ontoel
```

### Install Dependencies

```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch>=2.0.0

# Install other dependencies
pip install -r requirements.txt
```

<details>
<summary>📋 <b>Full Requirements</b></summary>

```
torch>=2.0.0
transformers>=4.30.0
faiss-cpu>=1.7.4      # or faiss-gpu for GPU support
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0
scikit-learn>=1.2.0
pyyaml>=6.0
scipy>=1.10.0
```

</details>

---

## 🚀 Quick Start

### 1️⃣ Prepare Data

```bash
# Use included sample data for testing
python scripts/preprocess_data.py --dataset sample --output_dir data/processed

# Or download and preprocess MedMentions
python scripts/download_datasets.py --dataset medmentions
python scripts/preprocess_data.py --dataset medmentions --output_dir data/processed
```

### 2️⃣ Train Model

```bash
# Quick test with sample data (~5 minutes)
python scripts/train.py --config configs/sample_config.yaml

# Full training on MedMentions (~2 hours on A100)
python scripts/train.py --config configs/medmentions_config.yaml
```

### 3️⃣ Evaluate

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --config configs/sample_config.yaml \
    --checkpoint checkpoints/best_model \
    --split test

# Interactive demo
python scripts/demo.py --checkpoint checkpoints/best_model
```

---

## 🏗️ Architecture

OntoEL operates in a **retrieve-then-reason** pipeline:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           OntoEL Pipeline                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  │   Mention   │     │   SapBERT   │     │  Candidate  │                │
│  │  + Context  │ ──▶ │   Encoder   │ ──▶ │  Retrieval  │                │
│  └─────────────┘     └─────────────┘     └──────┬──────┘                │
│                                                  │                       │
│                                                  ▼                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                │
│  │    Type     │     │   Fuzzy     │     │   Score     │                │
│  │  Inference  │ ──▶ │   Logic     │ ──▶ │   Fusion    │ ──▶ Output    │
│  │   τᴵ(m)     │     │   Layer     │     │  α·Sₙ+(1-α)·Sₒ│               │
│  └─────────────┘     └─────────────┘     └─────────────┘                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 🔬 Core Components

<details>
<summary><b>1. Neural Bi-Encoder</b></summary>

Uses SapBERT to encode mentions and entities into a shared semantic space:

```python
# Mention encoding
mention_emb = SapBERT([context_left, mention, context_right])

# Entity encoding  
entity_emb = SapBERT(entity_name)

# Neural similarity (cosine)
s_neural = cos_sim(mention_emb, entity_emb)
```

</details>

<details>
<summary><b>2. Context-Aware Type Inference</b></summary>

Infers semantic type memberships using dual projection:

```python
# Dual projection into type-inference space
m' = W_m @ mention_embedding      # Project mention
a'_τ = W_t @ type_name_embedding  # Project type name

# Fuzzy membership via scaled similarity
τᴵ(m) = σ(m' · a'_τ / θ)
```

</details>

<details>
<summary><b>3. Fuzzy Logic Layer</b></summary>

Computes ontological consistency using Product t-norm and Sigmoidal Reichenbach:

```python
# Sigmoidal Reichenbach Implication (resolves implication bias)
def I_σ(a, b, s=10):
    linear = 1 - a + a * b
    return sigmoid(s * (linear - 0.5))

# Consistency score (Product t-norm aggregation)
s_onto = Π_τ I_σ(τᴵ(m), τᴵ(e))
```

</details>

---

## ⚙️ Configuration

### Hyperparameters

| Parameter           | Default   | Description                   |
|:------------------- |:---------:|:----------------------------- |
| `encoder_name`      | `SapBERT` | Pretrained biomedical encoder |
| `hidden_dim`        | `768`     | Encoder hidden dimension      |
| `projection_dim`    | `768`     | Type projection dimension     |
| `sigmoid_sharpness` | `10`      | Sharpness (s) for I_σ         |
| `fusion_alpha`      | `0.8`     | Neural vs. logic balance      |
| `margin`            | `0.2`     | Ranking loss margin (γ)       |
| `type_loss_weight`  | `0.5`     | Type loss weight (λ)          |
| `learning_rate`     | `2e-5`    | AdamW learning rate           |
| `batch_size`        | `64`      | Training batch size           |
| `epochs`            | `10`      | Training epochs               |
| `top_k`             | `64`      | Candidates to retrieve        |

### Configuration Files

```yaml
# configs/medmentions_config.yaml
model:
  encoder_name: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  hidden_dim: 768
  projection_dim: 768

fuzzy_logic:
  sigmoid_sharpness: 10
  tnorm: "product"

fusion:
  alpha: 0.8

training:
  learning_rate: 2.0e-5
  batch_size: 64
  num_epochs: 10
```

---

## 📊 Results

### Main Results

<table>
<tr>
<th rowspan="2">Method</th>
<th colspan="3">MedMentions ST21pv</th>
<th colspan="3">BC5CDR</th>
<th colspan="3">NCBI-Disease</th>
</tr>
<tr>
<th>Acc@1</th><th>Acc@5</th><th>MRR</th>
<th>Acc@1</th><th>Acc@5</th><th>MRR</th>
<th>Acc@1</th><th>Acc@5</th><th>MRR</th>
</tr>
<tr>
<td>SapBERT</td>
<td>82.3</td><td>86.1</td><td>84.0</td>
<td>88.0</td><td>91.5</td><td>89.8</td>
<td>87.8</td><td>90.8</td><td>89.1</td>
</tr>
<tr>
<td>MedCPT</td>
<td>85.0</td><td>87.6</td><td>86.4</td>
<td>89.5</td><td>92.4</td><td>91.1</td>
<td>88.9</td><td>91.4</td><td>90.3</td>
</tr>
<tr style="background-color: #e8f5e9;">
<td><b>OntoEL (Ours)</b></td>
<td><b>87.8</b></td><td><b>88.9</b></td><td><b>88.2</b></td>
<td><b>90.5</b></td><td><b>93.1</b></td><td><b>91.9</b></td>
<td><b>89.8</b></td><td><b>92.0</b></td><td><b>91.0</b></td>
</tr>
</table>

### Ablation: Where Ontology Helps

```
┌────────────────────────────────────────────────────────────┐
│           Hard Subset Correction Rate                      │
│    (Mentions where backbone failed to rank gold @1)        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Cross-Encoder  ████████████████░░░░░░░░░░░░░░  40.2%     │
│  OntoEL         █████████████████████████████░  71.2%     │
│                                                            │
│  💡 OntoEL corrects 71% of hard ambiguous cases!          │
└────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
ontoel/
├── 📄 README.md
├── 📄 requirements.txt
├── 📁 assets/                   # Logo and figures
│   ├── ontoel_logo.svg
│   └── architecture_overview.png
├── 📁 configs/                  # Configuration files
│   ├── default_config.yaml
│   ├── sample_config.yaml
│   └── medmentions_config.yaml
├── 📁 data/
│   ├── 📁 sample/              # Sample data for testing
│   └── 📁 processed/           # Preprocessed datasets
├── 📁 src/                      # Source code
│   ├── __init__.py
│   ├── model.py                # 🧠 OntoEL model
│   ├── fuzzy_logic.py          # 🔮 Fuzzy EL++ operators
│   ├── type_inference.py       # 🎯 Type inference module
│   ├── dataset.py              # 📊 Data loading
│   ├── retrieval.py            # 🔍 FAISS retrieval
│   ├── trainer.py              # 🏋️ Training loop
│   └── utils.py                # 🛠️ Utilities
├── 📁 scripts/                  # Executable scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── preprocess_data.py
│   └── demo.py
└── 📁 checkpoints/             # Saved models
```

---

## 🔬 Key Innovations

### 1. Resolving Implication Bias

Traditional fuzzy implications suffer from gradient pathology. Our **Sigmoidal Reichenbach** implication provides:

- ✅ Non-vanishing gradients for all input pairs
- ✅ Exponential discrimination for hard negatives
- ✅ Numerical stability without division

```python
# Theorem 1: Gradient Non-Degeneracy
# ||∇I_σ(a,b)|| > 0 for all (a,b) ∈ (0,1)²
```

### 2. Zero-Shot Type Generalization

By encoding type names rather than fixed IDs, OntoEL generalizes to unseen types:

```
Zero-shot types:  Neural-only → 40.0%
                  OntoEL      → 82.0%  (+42.0% absolute!)
```

### 3. Robustness to Ontology Incompleteness

OntoEL maintains performance even with 80% of TBox axioms removed:

```
Axioms Removed    Accuracy
      0%          87.8%
     20%          87.5%
     50%          86.2%
     80%          83.8%  (still +1.5% over baseline!)
```

---

## 📖 Citation

If you find OntoEL useful in your research, please cite our paper:

```bibtex
@inproceedings{zhao2026ontoel,
  title     = {OntoEL: Neuro-Symbolic Biomedical Entity Linking with 
               Differentiable Fuzzy EL++ Reasoning},
  author    = {Zhao, Yizheng},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference 
               on Research and Development in Information Retrieval},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ for the biomedical NLP community</sub>
</p>

<p align="center">
  <a href="https://github.com/anonymous/ontoel">
    <img src="https://img.shields.io/badge/⭐_Star_this_repo-If_you_find_it_useful!-yellow?style=for-the-badge" alt="Star">
  </a>
</p>
