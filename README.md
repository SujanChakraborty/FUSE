# FUSE: Fast Semi-Supervised Node Embedding Learning via Structural and Label-Aware Optimization

> **Sujan Chakraborty, Rahul Bordoloi, Anindya Sengupta, Olaf Wolkenhauer, Saptarshi Bej**
>
> School of Data Science, IISER Thiruvananthapuram · University of Rostock · Texas A&M University

---

## Overview

FUSE is a fast, scalable semi-supervised node embedding framework designed for graphs where **node features are unavailable**. It jointly optimizes three complementary objectives:

1. **Unsupervised structure preservation** — a novel linear-time approximation of graph modularity that captures global community structure without spectral decomposition.
2. **Supervised regularization** — minimizes intra-class embedding variance among labeled nodes, ensuring class-discriminative clustering.
3. **Semi-supervised propagation** — refines unlabeled node embeddings via random-walk-based label spreading with attention-weighted similarity.

These are unified into a single iterative gradient ascent scheme with QR-based orthonormalization, yielding high-quality embeddings that are 5–7× faster to compute than comparable baselines such as Node2Vec and DeepWalk, while matching or exceeding their downstream classification performance.

---

## Table of Contents

- [Motivation](#motivation)
- [Method](#method)
  - [Modularity Gradient Approximation](#modularity-gradient-approximation)
  - [Supervised Component](#supervised-component)
  - [Semi-Supervised Component](#semi-supervised-component)
  - [Unified Optimization](#unified-optimization)
  - [Computational Complexity](#computational-complexity)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Hyperparameters](#hyperparameters)
- [Experiments](#experiments)
  - [Datasets](#datasets)
  - [Baselines](#baselines)
  - [Results](#results)
  - [Runtime](#runtime)
- [Ablation Study](#ablation-study)
- [Scalability](#scalability)
- [Feature Integration](#feature-integration)
- [GPU Support](#gpu-support)
- [Extensions and Limitations](#extensions-and-limitations)
- [Citation](#citation)
- [License](#license)

---

## Motivation

In many real-world graphs — dynamic networks, protein interaction networks, privacy-sensitive domains, and recommendation systems — **node feature vectors are absent or unavailable**. Existing approaches (GCN, GAT, GraphSAGE) initialized with random embeddings perform poorly in this regime. Random-walk methods like Node2Vec and DeepWalk produce strong embeddings but are computationally expensive. FUSE fills this gap: it generates structure- and label-aware embeddings from scratch with no precomputed features, in a fraction of the time.

---

## Method

### Modularity Gradient Approximation

FUSE optimizes a differentiable form of graph modularity:

$Q(S) = \frac{1}{2m} \mathbb{Tr}(S^\top B S), \quad B = A - \frac{dd^\top}{2m}$

where $S \in \mathbb{R}^{n \times k}$ is the embedding matrix, $A$ is the adjacency matrix, $d$ is the degree vector, and $m = |E|$.

Instead of the exact gradient $\nabla_S Q = \frac{1}{m}(AS - \frac{1}{2m}d(d^\top S))$, FUSE uses a stable approximation:

$$\nabla_S Q_{\text{prop}} = \frac{1}{2m}\left(AS - \frac{1}{2m} d(\mathbf{1}^\top S)\right)$$

replacing the degree-weighted mean $d^\top S$ with the unweighted global mean $\mathbf{1}^\top S$. This reduces hub dominance and is provably directionally stable (cosine similarity ≥ 0.78 on small graphs, ≥ 0.99 on large ones — see Theorem 1 in the paper).

### Supervised Component

For labeled nodes, FUSE minimizes intra-class embedding variance:

$$Q_{\text{sup}} = \sum_c \sum_{i \in C_c} \|S_{i,:} - \mu_c\|^2, \quad \mu_c = \frac{1}{|C_c|}\sum_{i \in C_c} S_{i,:}$$

Gradient: $\nabla Q_{\text{sup}} = S - \tilde{S}$, where $\tilde{S}_i = \mu_c$ for $i \in C_c$.

### Semi-Supervised Component

For unlabeled nodes, FUSE runs **label-biased random walks** that preferentially visit labeled nodes (up to $L'$ labeled steps per walk). Visited labeled nodes per unlabeled node $i$ are collected into a set $\rho(i)$. Attention weights are computed as:

$$w_{ij} = \frac{\exp(S_{i,:}^\top S_{j,:})}{\sum_{k \in \rho(i)} \exp(S_{i,:}^\top S_{k,:})}$$

The semi-supervised gradient pulls unlabeled nodes toward their attention-weighted labeled neighbors:

$$\nabla_S Q_{\text{semi}} = S_{i,:} - \sum_j w_{ij} S_{j,:}$$

### Unified Optimization

All three gradients are combined in a gradient ascent step:

$$\nabla_S Q_{\text{total}} = \nabla_S Q_{\text{prop}} - \lambda_{\text{sup}} \nabla_S Q_{\text{sup}} - \lambda_{\text{semi}} \nabla_S Q_{\text{semi}}$$

$$S \leftarrow S + \eta \nabla_S Q_{\text{total}}$$

After each iteration, $S$ is orthonormalized via QR decomposition to maintain stability.

### Computational Complexity

| Component | Cost |
|---|---|
| Modularity gradient ($AS$, degree correction) | $O(|E|k + nk)$ |
| Supervised gradient | $O(nk)$ |
| Random walks | $O(w \ell)$ |
| Attention updates | $O(n d_{\max} k)$ |
| QR orthonormalization | $O(nk^2)$ |
| **Total (per iteration)** | **$O(|E|k + nk + nd_{\max}k + w\ell + nk^2)$** |

This is substantially better than spectral methods requiring $O(n^3)$ eigendecomposition.

---

## Installation

```bash
git clone https://github.com/<your-org>/FUSE.git
cd FUSE
pip install -r requirements.txt
```

**Dependencies:**
- Python ≥ 3.8
- NumPy
- SciPy
- PyTorch (optional, for GPU variant)
- PyTorch Geometric (optional, for GPU variant)
- scikit-learn
- networkx

All experiments in the paper were run on a 13th Gen Intel Core i9-13900 CPU with 64 GB RAM, **no GPU**, unless stated otherwise (see GPU experiments in Appendix B.6).

---

## Usage

### Basic Example

```python
from fuse import FUSE

# G: networkx graph or adjacency matrix
# labels: array of node labels (-1 for unlabeled)
model = FUSE(
    k=150,          # embedding dimension
    eta=0.05,       # learning rate
    lambda_sup=1.0, # supervised loss weight
    lambda_semi=2.0,# semi-supervised loss weight
    T=200,          # number of iterations
    r=10,           # random walks per node
    L=5,            # walk length
    L_prime=3       # max labeled steps per walk
)

embeddings = model.fit(G, labels)
```

The returned `embeddings` array (shape `[n, k]`) can be used directly as input to any downstream GNN classifier (GCN, GAT, GraphSAGE) or MLP.

### Hyperparameters

The default hyperparameters used across all datasets in the paper:

| Parameter | Value | Description |
|---|---|---|
| `k` | 150 | Embedding dimension |
| `eta` (η) | 0.05 | Learning rate |
| `lambda_sup` (λ_sup) | 1.0 | Weight of supervised loss |
| `lambda_semi` (λ_semi) | 2.0 | Weight of semi-supervised loss |
| `T` | 200 | Gradient ascent iterations |
| `r` | 10 | Random walks per node |
| `L` | 5 | Walk length |
| `L'` | 3 | Max labeled steps per walk |

**Sensitivity notes** (from Appendix B.4):
- `eta`, `lambda_sup`, `lambda_semi` are the most sensitive parameters and benefit from dataset-specific tuning.
- `r`, `L`, `L'` are more robust; moderate values work well across datasets.
- Larger `T` or `L` can yield marginal accuracy gains at higher runtime cost.

Optimal hyperparameters found via search (30-70 split):

| Dataset | k | η | λ_sup | λ_semi | T | r | L | L' |
|---|---|---|---|---|---|---|---|---|
| Cora | 145 | 0.31 | 0.6 | 1.9 | 200 | 20 | 4 | 1 |
| CiteSeer | 135 | 0.51 | 0.8 | 1.5 | 450 | 13 | 5 | 3 |
| PubMed | 155 | 0.11 | 0.9 | 2.0 | 450 | 12 | 9 | 1 |
| WikiCS | 130 | 0.28 | 1.1 | 1.1 | 200 | 20 | 3 | 2 |
| Amazon Photo | 100 | 0.21 | 1.7 | 2.5 | 300 | 13 | 3 | 2 |

---

## Experiments

### Datasets

| Dataset | Nodes | Edges | Classes | Feature Dim |
|---|---|---|---|---|
| Cora | 2,708 | 5,429 | 7 | 1,433 |
| CiteSeer | 3,327 | 9,104 | 6 | 3,703 |
| PubMed | 19,717 | 44,338 | 3 | 500 |
| Amazon Photo | 7,487 | 119,043 | 8 | 745 |
| WikiCS | 11,701 | 216,123 | 10 | 300 |
| arXiv | 169,343 | 1,166,243 | 40 | 128 |

All experiments assume **node features are unavailable** (embeddings are generated from scratch), except for feature integration experiments (Appendix B.7).

### Baselines

**Unsupervised:** Node2Vec, DeepWalk, VGAE, M-NMF

**Self-supervised:** DGI, COLES, CCA-SSG, MVGRL

**Semi-supervised:** GraFN, ReVAR

**Trivial:** Random embeddings (lower bound), given features (upper bound)

All embeddings are evaluated with three GNN classifiers: **GCN**, **GAT**, **GraphSAGE**.

### Results

Selected classification results averaged across all datasets (70-30 split):

| Classifier | Method | Accuracy | F1 |
|---|---|---|---|
| GAT | FUSE | 0.82 | 0.80 |
| GAT | DeepWalk | 0.82 | 0.80 |
| GAT | Node2Vec | 0.82 | 0.80 |
| GAT | DGI | 0.59 | 0.51 |
| **GCN** | **FUSE** | **0.78** | **0.76** |
| GCN | DeepWalk | 0.64 | 0.58 |
| GCN | Node2Vec | 0.64 | 0.57 |
| SAGE | FUSE | 0.80 | 0.77 |
| SAGE | DeepWalk | 0.81 | 0.79 |

FUSE particularly shines with **GCN**, where it substantially outperforms Node2Vec and DeepWalk while being much faster. Full per-dataset results are in Tables 37–41 of the paper.

**Clustering performance:** FUSE achieves the highest V-Measure scores across all six datasets (GAT classifier), indicating superior alignment of learned embeddings with ground-truth class labels.

### Runtime

Average embedding generation time (seconds, 5 runs):

| Method | Cora | CiteSeer | Photo | WikiCS | PubMed | Average |
|---|---|---|---|---|---|---|
| FUSE | 12.5 | 13.4 | 49.5 | 86.5 | 95.8 | 51.5 |
| DeepWalk | 50.5 | 51.4 | 292.3 | 747.2 | 490.7 | 326.4 |
| Node2Vec | 47.3 | 50.3 | 288.3 | 745.3 | 453.7 | 317.0 |
| VGAE | 13.0 | 14.3 | 137.3 | 329.5 | 235.2 | 145.9 |
| MVGRL | 516.4 | 607.6 | 559.3 | 718.9 | 1241.2 | 733.2 |

FUSE is approximately **5× faster** than Node2Vec/DeepWalk on average, and **7× faster** on large-scale graphs (arXiv: FUSE 1,360s vs DeepWalk ~13,000s with L=5).

---

## Ablation Study

Three variants are evaluated: unsupervised-only (modularity), semi-supervised-only, and the full model (both + supervised). Key findings:

- The **unsupervised modularity component alone** performs strongly, especially for GraphSAGE, confirming that community structure provides a powerful inductive bias even without labels.
- **Combining all three components** consistently yields the best accuracy and F1 overall.
- The runtime overhead of adding semi-supervised propagation is marginal (~10–20% over unsupervised-only).

Average classification accuracy across datasets (30-70 split):

| Variant | GAT | GCN | SAGE |
|---|---|---|---|
| Unsupervised only | 0.688 | 0.660 | 0.716 |
| Semi-supervised only | 0.690 | 0.656 | 0.525 |
| **Full model** | **0.697** | **0.668** | **0.732** |

---

## Scalability

FUSE has been evaluated on three large-scale graphs:

| Dataset | Nodes | Edges | FUSE Time | DeepWalk Time |
|---|---|---|---|---|
| arXiv | ~169K | ~1.2M | ~1,360s | ~13,000s |
| MAG | ~736K | ~8M | ~4,076s | ~5,549s |
| Products | ~2.45M | ~61.9M | ~36,571s | > 24 hours |

The **unsupervised variant** is even faster (e.g., ~1,520s on MAG, ~10,335s on Products) and is recommended when speed is the primary concern.

**Label masking robustness:** FUSE was tested under MCAR, MAR, and MNAR masking at rates from 1% to 80%, showing consistent competitive performance — particularly strong at very low (1–5%) and high (80%) label availability.

---

## Feature Integration

When node features are available, FUSE can incorporate them via two strategies:

**S1 — Feature Reconstruction Gradient:** Adds a fourth gradient term penalizing reconstruction error $\|SW - X\|_F^2$. Particularly effective for text-attributed graphs (CiteSeer, PubMed, WikiCS).

**S2 — Feature-Augmented Adjacency:** Constructs a hybrid adjacency $A_{\text{hybrid}} = (1-\beta)A_{\text{structural}} + \beta A_{\text{feature-kNN}}$ before running FUSE unchanged. Effective when feature neighborhoods reveal community structure missing from the graph topology (Cora, Amazon Photo).

Best results (70% labels):

| Dataset | Strategy | Classifier | Accuracy | F1 |
|---|---|---|---|---|
| Cora | S2 | GAT | 0.873 | 0.860 |
| CiteSeer | S1 | GAT | 0.733 | 0.702 |
| PubMed | S1 | SAGE | 0.885 | 0.882 |
| Amazon Photo | S2 | GAT | 0.925 | 0.916 |
| WikiCS | S1 | GAT | 0.820 | 0.798 |

---

## GPU Support

A GPU-accelerated variant is available (Appendix B.6). Core operations (sparse adjacency multiplication, gradient computation, QR orthonormalization) are implemented via PyTorch sparse operations and GPU-based QR. Random walk generation remains on CPU (inherently sequential).

GPU runtimes (seconds):

| Method | Cora | CiteSeer | Photo | WikiCS | PubMed |
|---|---|---|---|---|---|
| FUSE (GPU) | 3.0 | 2.9 | 9.7 | 17.0 | 14.6 |
| DGI (GPU) | 4.1 | 4.1 | 41.9 | 74.2 | 25.5 |
| VGAE (GPU) | 8.5 | 9.4 | 108.8 | 209.3 | 253.0 |
| DeepWalk | 57.6 | 60.4 | 544.0 | 785.1 | 511.8 |

Even under GPU settings, FUSE achieves **orders-of-magnitude lower runtimes** than DeepWalk/Node2Vec while maintaining competitive accuracy.

---

## Extensions and Limitations

**Limitations:**
- FUSE assumes node features are unavailable; while features can be incorporated via S1/S2, the core method is designed for the feature-absent regime.
- The method is primarily designed for **homophilous graphs**, where class labels align with community structure. Adaptation to heterophilous graphs (e.g., Chameleon, Squirrel) requires modifications to the objective.
- On very large graphs (MAG, Products), there is a trade-off between speed and accuracy compared to DeepWalk.

**Future directions:**
- Adapting FUSE to dynamically evolving graphs.
- Extending the objective to handle heterophily.
- Incorporating incremental label updates for streaming settings.

---

## Citation

If you use FUSE in your research, please cite:

```bibtex
@article{chakraborty2025fuse,
  title     = {FUSE: Fast Semi-Supervised Node Embedding Learning via Structural and Label-Aware Optimization},
  author    = {Chakraborty, Sujan and Bordoloi, Rahul and Sengupta, Anindya and Wolkenhauer, Olaf and Bej, Saptarshi},
  journal   = {Neural Networks},
  year      = {2025},
  note      = {Manuscript ID: NEUNET-D-26-03959}
}
```

---

## Acknowledgements

O.W. acknowledges support from the German Research Foundation (DFG) FK515800538 (*Learning convex data spaces*).

---

## License

This repository is released for academic research use. Please refer to the `LICENSE` file for details.
