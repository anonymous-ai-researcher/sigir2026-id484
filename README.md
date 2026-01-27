<p align="center">
  <img src="assets/ontoel_logo.svg" alt="OntoEL Logo" width="400"/>
</p>

<h1 align="center">🧬 OntoEL</h1>

<p align="center">
  <strong>Neuro-Symbolic Biomedical Entity Linking with Differentiable Fuzzy EL⊥ Reasoning</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-extended-appendix">📚 Appendix</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-results">Results</a> 
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

<table>
<tr>
<td>

## 📢 Important Note to Reviewers

This repository accompanies our submission and includes a **comprehensive 10-section appendix** with:

✅ **Complete theoretical proofs** for all theorems (Gradient Non-Degeneracy, Semantic Soundness)  
✅ **Zero-shot generalization bounds** with 7 lemmas establishing Lipschitz continuity  
✅ **Score fusion optimality analysis** proving Bayes-optimality of linear combination  
✅ **Extensive ablation studies** on fuzzy implications, sharpness, and type loss weight  
✅ **Per-type performance breakdown** across all 21 semantic types  
✅ **Detailed error analysis** with taxonomy and case studies  

<p align="center">
  <b>👉 Please see <a href="#-extended-appendix">Section: 📚 Extended Appendix</a> for the complete structure and key findings.</b>
</p>

</td>
</tr>
</table>

---

## 🎯 Overview

**OntoEL** is a state-of-the-art neuro-symbolic framework that bridges the gap between neural retrieval and logical reasoning for biomedical entity linking. Unlike traditional approaches that treat ontologies as flat dictionaries, OntoEL leverages the rich terminological knowledge (TBox) encoded in biomedical ontologies through differentiable fuzzy Description Logic.

<p align="center">
  <img src="assets/architecture_overview.svg" alt="OntoEL Architecture" width="800"/>
</p>

### ✨ Key Features

<table>
<thead>
<tr>
<th align="left">Feature</th>
<th align="left">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>🧠 <b>Neuro-Symbolic Fusion</b></td>
<td>Combines neural bi-encoder retrieval with fuzzy EL⊥ reasoning</td>
</tr>
<tr>
<td>🎯 <b>Context-Aware Typing</b></td>
<td>Infers semantic types from mention context using dual projection</td>
</tr>
<tr>
<td>⚡ <b>Gradient-Stable Logic</b></td>
<td>Sigmoidal Reichenbach implication resolves "implication bias"</td>
</tr>
<tr>
<td>🔄 <b>End-to-End Training</b></td>
<td>Ontological axioms as differentiable soft constraints</td>
</tr>
<tr>
<td>🚀 <b>Efficient Inference</b></td>
<td>30x faster than cross-encoders with comparable accuracy</td>
</tr>
</tbody>
</table>

### 🏆 Performance Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOTA Results on BioEL Benchmarks             │
├─────────────────────────────────────────────────────────────────┤
│  📊 MedMentions ST21pv    │  Acc@1: 86.5%  │  +4.2% vs baseline │
│  💊 BC5CDR                │  Acc@1: 90.5%  │  +2.5% vs baseline │
│  🏥 NCBI-Disease          │  Acc@1: 89.8%  │  +1.9% vs baseline │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 Extended Appendix

> **The extended version contains 10 appendix sections that provide comprehensive theoretical foundations, ablation studies, and detailed analyses supporting all claims in the main paper.**

### 📖 Complete Appendix Structure

<table>
<thead>
<tr>
<th align="center">§</th>
<th align="left">Title</th>
<th align="left">Content</th>
<th align="left">Key Findings</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center"><b>A</b></td>
<td>Theoretical Proofs</td>
<td>Complete proofs for Theorem 1 (Gradient Non-Degeneracy) and Theorem 2 (Semantic Soundness)</td>
<td>Sigmoidal Reichenbach guarantees ∇≠0 for all (a,b)∈(0,1)²; DR grows as ~e<sup>0.31s</sup></td>
</tr>
<tr>
<td align="center"><b>B</b></td>
<td>Complexity Analysis</td>
<td>Formal analysis of Proposition 1 (Inference Efficiency)</td>
<td>Logic module adds <5% latency (~4ms overhead)</td>
</tr>
<tr>
<td align="center"><b>C</b></td>
<td>Zero-Shot Generalization</td>
<td>7 lemmas/theorems establishing Lipschitz bounds for type inference</td>
<td>Membership error ≤ L·‖a<sub>τ₁</sub> - a<sub>τ₂</sub>‖₂ enables transfer to unseen types</td>
</tr>
<tr>
<td align="center"><b>D</b></td>
<td>Score Fusion Optimality</td>
<td>3 propositions proving Bayes-optimality of linear fusion</td>
<td>α*≈0.8 reflects 4:1 SNR ratio; robust in [0.6, 0.9]</td>
</tr>
<tr>
<td align="center"><b>E</b></td>
<td>Type Constraint Comparison</td>
<td>Comparison with 4 baselines: Hard Filter, Binary Match, Hinge Loss, Fuzzy DL</td>
<td>Fuzzy EL⊥ provides +0.6–1.1% over alternatives</td>
</tr>
<tr>
<td align="center"><b>F</b></td>
<td>Fuzzy Implication Ablation</td>
<td>Goguen, Łukasiewicz, Reichenbach, Sigmoidal (s=3,5,8,10,15,20)</td>
<td>Sigmoidal outperforms classical by +1.6–3.3%; optimal s∈[8,15]</td>
</tr>
<tr>
<td align="center"><b>G</b></td>
<td>Per-Type Performance</td>
<td>Breakdown across all 21 UMLS semantic types</td>
<td>Disorder/Finding gain +7.3–7.4%; disjointness axiom accounts for 65% corrections</td>
</tr>
<tr>
<td align="center"><b>H</b></td>
<td>Error Analysis</td>
<td>Taxonomy of 200 sampled errors with detailed case studies</td>
<td>Type inference errors (33.5%) are primary bottleneck; correction ratio = 2.0:1</td>
</tr>
<tr>
<td align="center"><b>I</b></td>
<td>Data Construction</td>
<td>UMLS → EL⊥ TBox mapping protocol</td>
<td>28,651 subsumption + 45 disjointness axioms for MedMentions</td>
</tr>
<tr>
<td align="center"><b>J</b></td>
<td>Implementation Details</td>
<td>Full hyperparameters, baseline configurations, compute costs</td>
<td>All settings for complete reproducibility</td>
</tr>
</tbody>
</table>

---

### 🔑 Key Theoretical Results

#### 📐 Theorem 1: Gradient Non-Degeneracy (Appendix A)

The Sigmoidal Reichenbach implication resolves the **implication bias** problem that plagues classical fuzzy implications:

<table>
<thead>
<tr>
<th align="left">Property</th>
<th align="center">Goguen</th>
<th align="center">Łukasiewicz</th>
<th align="center">Reichenbach</th>
<th align="center"><b>Sigmoidal (Ours)</b></th>
</tr>
</thead>
<tbody>
<tr>
<td>Non-zero gradient region</td>
<td align="center">50%</td>
<td align="center">50%</td>
<td align="center">100%</td>
<td align="center"><b>100%</b></td>
</tr>
<tr>
<td>Adaptive gradient magnitude</td>
<td align="center">✗</td>
<td align="center">✗</td>
<td align="center">✗</td>
<td align="center"><b>✓</b></td>
</tr>
<tr>
<td>Discrimination Ratio (s=10)</td>
<td align="center">∞*</td>
<td align="center">1.0</td>
<td align="center">5.0</td>
<td align="center"><b>23.0</b></td>
</tr>
</tbody>
</table>

<sub>*Goguen has infinite DR in theory but zero gradients in 50% of the domain prevent learning.</sub>

**Key Result:** Discrimination Ratio grows as ~e<sup>0.31s</sup>, providing exponential advantage over linear Reichenbach.

---

#### 📊 Fuzzy Implication & Sharpness Ablation (Appendix F)

Comprehensive ablation on MedMentions ST21pv with SapBERT backbone:

<table>
<thead>
<tr>
<th align="left">Implication</th>
<th align="center">Sharpness s</th>
<th align="center">Acc@1</th>
<th align="center">Δ vs Baseline</th>
</tr>
</thead>
<tbody>
<tr>
<td>None (SapBERT only)</td>
<td align="center">—</td>
<td align="center">82.3%</td>
<td align="center">—</td>
</tr>
<tr>
<td>Goguen</td>
<td align="center">—</td>
<td align="center">83.2%</td>
<td align="center">+0.9%</td>
</tr>
<tr>
<td>Łukasiewicz</td>
<td align="center">—</td>
<td align="center">83.6%</td>
<td align="center">+1.3%</td>
</tr>
<tr>
<td>Reichenbach</td>
<td align="center">—</td>
<td align="center">84.9%</td>
<td align="center">+2.6%</td>
</tr>
<tr style="background-color: #e8f5e9;">
<td><b>Sigmoidal</b></td>
<td align="center"><b>s=10</b></td>
<td align="center"><b>86.5%</b></td>
<td align="center"><b>+4.2%</b></td>
</tr>
</tbody>
</table>

**Sharpness Sensitivity:** Optimal range s∈[8, 15]; s=10 balances discrimination (DR≈23) with gradient stability.

---

#### 🎯 Zero-Shot Generalization Bounds (Appendix C)

**Theorem (Semantic Continuity):** For any mention m and semantic types τ₁, τ₂:

```
|τ₁ᴵ(m) - τ₂ᴵ(m)| ≤ (B_m · B_t · ‖m‖₂) / (4θ) · ‖a_τ₁ - a_τ₂‖₂
```

**Implication:** Semantically similar type names yield similar membership predictions, enabling zero-shot transfer to unseen types without retraining.

**Empirical Validation:** +41.3pp improvement on zero-shot types (SapBERT: 35.2% → OntoEL: 76.5%)

---

#### ⚖️ Score Fusion Optimality (Appendix D)

**Proposition (Bayes-Optimal Linear Fusion):** Under conditional independence assumption:

```
α* = SNR_neural / (SNR_neural + SNR_onto)
```

**Key Findings:**

- α ≈ 0.8 implies SNR<sub>neural</sub> / SNR<sub>onto</sub> ≈ 4:1
- Performance plateau in [0.6, 0.9] due to score correlation
- Linear fusion outperforms max/product/MLP alternatives

---

#### 🔍 Error Analysis Summary (Appendix H)

**Error Taxonomy (200 sampled errors):**

<table>
<thead>
<tr>
<th align="left">Error Type</th>
<th align="center">%</th>
<th align="left">Dominant Semantic Types</th>
</tr>
</thead>
<tbody>
<tr>
<td>Type Inference Error</td>
<td align="center">33.5%</td>
<td>Disorder, Finding, Clinical Attribute</td>
</tr>
<tr>
<td>Ontology Incompleteness</td>
<td align="center">21.0%</td>
<td>Anatomical Structure</td>
</tr>
<tr>
<td>Surface Form Ambiguity</td>
<td align="center">19.0%</td>
<td>Chemical, Gene</td>
</tr>
<tr>
<td>Context Insufficiency</td>
<td align="center">15.5%</td>
<td>Health Care Activity</td>
</tr>
<tr>
<td>Candidate Recall Failure</td>
<td align="center">11.0%</td>
<td>—</td>
</tr>
</tbody>
</table>

**Error Flow Analysis:**

<table>
<thead>
<tr>
<th align="left">Category</th>
<th align="center">Count</th>
<th align="center">%</th>
</tr>
</thead>
<tbody>
<tr>
<td>Both Correct</td>
<td align="center">10,264</td>
<td align="center">78.1%</td>
</tr>
<tr>
<td>Both Wrong</td>
<td align="center">1,222</td>
<td align="center">9.3%</td>
</tr>
<tr>
<td>Corrected by OntoEL</td>
<td align="center">1,104</td>
<td align="center">8.4%</td>
</tr>
<tr>
<td>Introduced by OntoEL</td>
<td align="center">552</td>
<td align="center">4.2%</td>
</tr>
<tr style="background-color: #e8f5e9;">
<td><b>Net Improvement</b></td>
<td align="center"><b>+552</b></td>
<td align="center"><b>+4.2%</b></td>
</tr>
</tbody>
</table>

**Correction Ratio = 2.0:1** — OntoEL corrects twice as many errors as it introduces.

---

#### 📈 Per-Type Performance Highlights (Appendix G)

**Top Improvements (Δ > 5%):**

<table>
<thead>
<tr>
<th align="left">Semantic Type</th>
<th align="center">SapBERT</th>
<th align="center">OntoEL</th>
<th align="center">Δ</th>
<th align="left">Primary Confusion</th>
</tr>
</thead>
<tbody>
<tr>
<td>Disorder</td>
<td align="center">78.2%</td>
<td align="center">85.6%</td>
<td align="center"><b>+7.4%</b></td>
<td>Finding, Procedure</td>
</tr>
<tr>
<td>Finding</td>
<td align="center">75.8%</td>
<td align="center">83.1%</td>
<td align="center"><b>+7.3%</b></td>
<td>Disorder</td>
</tr>
<tr>
<td>Injury or Poisoning</td>
<td align="center">76.5%</td>
<td align="center">83.2%</td>
<td align="center"><b>+6.7%</b></td>
<td>Disorder, Procedure</td>
</tr>
<tr>
<td>Body Substance</td>
<td align="center">79.3%</td>
<td align="center">85.4%</td>
<td align="center"><b>+6.1%</b></td>
<td>Chemical</td>
</tr>
<tr>
<td>Biologic Function</td>
<td align="center">77.8%</td>
<td align="center">83.4%</td>
<td align="center"><b>+5.6%</b></td>
<td>Finding</td>
</tr>
</tbody>
</table>

**Disjointness Axiom Impact:** The single axiom `Disorder ⊓ Finding ⊑ ⊥` accounts for **65%** of all corrections from disjointness reasoning.

---

### ✅ Reviewer Checklist

We recommend reviewers examine the following sections for key claims:

- [ ] **Appendix A:** Theorem 1 proof (gradient non-degeneracy for Sigmoidal Reichenbach)
- [ ] **Appendix C:** Zero-shot generalization bounds (7 lemmas with Lipschitz analysis)
- [ ] **Appendix D:** Score fusion optimality (Bayes-optimal linear combination)
- [ ] **Appendix F, Table 7:** Implication & sharpness ablation results
- [ ] **Appendix F, Table 8:** Type loss weight (λ) sensitivity analysis
- [ ] **Appendix G, Table 9:** Per-type performance breakdown (21 types)
- [ ] **Appendix H, Table 10:** Error flow analysis with correction ratio

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
│                           OntoEL Pipeline                                │
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
│  │   τᴵ(m)     │     │   Layer     │     │ α·Sₙ+(1-α)·Sₒ│               │
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

<table>
<thead>
<tr>
<th align="left">Parameter</th>
<th align="center">Default</th>
<th align="left">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>encoder_name</code></td>
<td align="center">SapBERT</td>
<td>Pretrained biomedical encoder</td>
</tr>
<tr>
<td><code>hidden_dim</code></td>
<td align="center">768</td>
<td>Encoder hidden dimension</td>
</tr>
<tr>
<td><code>projection_dim</code></td>
<td align="center">768</td>
<td>Type projection dimension</td>
</tr>
<tr>
<td><code>sigmoid_sharpness</code></td>
<td align="center">10</td>
<td>Sharpness (s) for I_σ</td>
</tr>
<tr>
<td><code>fusion_alpha</code></td>
<td align="center">0.8</td>
<td>Neural vs. logic balance</td>
</tr>
<tr>
<td><code>margin</code></td>
<td align="center">0.2</td>
<td>Ranking loss margin (γ)</td>
</tr>
<tr>
<td><code>type_loss_weight</code></td>
<td align="center">0.5</td>
<td>Type loss weight (λ)</td>
</tr>
<tr>
<td><code>learning_rate</code></td>
<td align="center">2e-5</td>
<td>AdamW learning rate</td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td align="center">64</td>
<td>Training batch size</td>
</tr>
<tr>
<td><code>epochs</code></td>
<td align="center">10</td>
<td>Training epochs</td>
</tr>
<tr>
<td><code>top_k</code></td>
<td align="center">64</td>
<td>Candidates to retrieve</td>
</tr>
</tbody>
</table>

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
<thead>
<tr>
<th rowspan="2" align="left">Method</th>
<th colspan="3" align="center">MedMentions ST21pv</th>
<th colspan="3" align="center">BC5CDR</th>
<th colspan="3" align="center">NCBI-Disease</th>
</tr>
<tr>
<th align="center">Acc@1</th>
<th align="center">Acc@5</th>
<th align="center">MRR</th>
<th align="center">Acc@1</th>
<th align="center">Acc@5</th>
<th align="center">MRR</th>
<th align="center">Acc@1</th>
<th align="center">Acc@5</th>
<th align="center">MRR</th>
</tr>
</thead>
<tbody>
<tr>
<td>SapBERT</td>
<td align="center">82.3</td>
<td align="center">86.1</td>
<td align="center">84.0</td>
<td align="center">88.0</td>
<td align="center">91.5</td>
<td align="center">89.8</td>
<td align="center">87.8</td>
<td align="center">90.8</td>
<td align="center">89.1</td>
</tr>
<tr>
<td>MedCPT</td>
<td align="center">85.0</td>
<td align="center">87.6</td>
<td align="center">86.4</td>
<td align="center">89.5</td>
<td align="center">92.4</td>
<td align="center">91.1</td>
<td align="center">88.9</td>
<td align="center">91.4</td>
<td align="center">90.3</td>
</tr>
<tr style="background-color: #e8f5e9;">
<td><b>OntoEL (Ours)</b></td>
<td align="center"><b>86.5</b></td>
<td align="center"><b>88.9</b></td>
<td align="center"><b>87.6</b></td>
<td align="center"><b>90.5</b></td>
<td align="center"><b>93.1</b></td>
<td align="center"><b>91.9</b></td>
<td align="center"><b>89.8</b></td>
<td align="center"><b>92.0</b></td>
<td align="center"><b>91.0</b></td>
</tr>
</tbody>
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
│   ├── fuzzy_logic.py          # 🔮 Fuzzy EL⊥ operators
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
Zero-shot types:  Neural-only → 35.2%
                  OntoEL      → 76.5%  (+41.3% absolute!)
```

### 3. Robustness to Ontology Incompleteness

OntoEL maintains performance even with 80% of TBox axioms removed:

```
Axioms Removed    Accuracy
      0%          86.5%
     20%          86.2%
     50%          85.1%
     80%          83.8%  (still +1.5% over baseline!)
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
