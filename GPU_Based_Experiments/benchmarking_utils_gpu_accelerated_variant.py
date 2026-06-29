# benchmarking_utils.py
"""
Utilities for node embedding benchmarking across datasets:
- dataset loaders (Cora, CiteSeer, PubMed, WikiCS, Amazon-Photo)
- embedding generators: deepwalk, node2vec, vgae, dgi, fuse (FUSE = semi-supervised modularity),
  random, given (projected to 150 dim if needed)
- classifiers: GCN, GAT, GraphSAGE (Spektral)
- training, evaluation, saving utilities
- run_benchmark() orchestrator
"""
import os
import time
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import lil_matrix, csr_matrix
import scipy.linalg
import numpy as np

if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu
# embedding libs
from node2vec import Node2Vec
import torch
from torch_geometric.nn import DeepGraphInfomax, VGAE as PyG_VGAE
from torch_geometric.nn import GCNConv as PyG_GCNConv
from torch_geometric.data import Data as PyGData

# Spektral + TF for classifiers (used in your original code)
import tensorflow as tf
from spektral.layers import GCNConv, GATConv, GraphSageConv
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import TruncatedSVD

# Torch-geo datasets + Spektral/similar dataset imports
from torch_geometric.datasets import Planetoid, WikiCS, Amazon
from spektral.datasets import Cora

import logging
logging.getLogger("gensim").setLevel(logging.ERROR)

# Suppress Python warnings
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configurable default
# ----------------------------
DEFAULT_EMB_DIM = 150

# ----------------------------
# Dataset loaders (preserve original loading code)
# ----------------------------
def load_dataset(dataset_name, root="."):
    """
    Load dataset by name. Returns a dict:
    {
      'x': numpy array features,
      'a': scipy sparse adjacency (csr),
      'y': one-hot labels (numpy),
      'labels': integer labels (numpy),
      'G': networkx Graph,
      'pyg_data': torch_geometric.data.Data (x tensor, edge_index, y tensor)
    }
    dataset_name options: 'cora', 'citeseer', 'pubmed', 'wikics', 'photo' (amazon-photo)
    """
    name = dataset_name.lower()
    if name == "cora":
        data = Cora()
        graph = data.graphs[0]
        x = graph.x
        a = graph.a.tocsr() if sp.issparse(graph.a) else csr_matrix(graph.a)
        y_onehot = graph.y
        labels = np.argmax(y_onehot, axis=1)
    elif name in ("citeseer", "pubmed"):
        # Planetoid provides CiteSeer & PubMed
        data = Planetoid(root=root, name=dataset_name.capitalize())
        d = data[0]
        x = d.x.numpy()
        edge_index = d.edge_index.numpy()
        labels = d.y.numpy()
        num_nodes = x.shape[0]
        a = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for i in range(edge_index.shape[1]):
            s, t = edge_index[:, i]
            a[s, t] = 1
            a[t, s] = 1
        a = a.tocsr()
        num_classes = labels.max() + 1
        y_onehot = np.eye(num_classes)[labels]
    elif name == "wikics":
        data = WikiCS(root=root)
        d = data[0]
        x = d.x.numpy()
        edge_index = d.edge_index.numpy()
        labels = d.y.numpy()
        num_nodes = x.shape[0]
        a = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for i in range(edge_index.shape[1]):
            s, t = edge_index[:, i]
            a[s, t] = 1
            a[t, s] = 1
        a = a.tocsr()
        num_classes = labels.max() + 1
        y_onehot = np.eye(num_classes)[labels]
    elif name in ("photo", "amazon-photo", "amazon_photos", "amazon_photos"):
        # torch_geometric Amazon dataset (photo)
        data = Amazon(root=root, name="photo")
        d = data[0]
        x = d.x.numpy()
        edge_index = d.edge_index.numpy()
        labels = d.y.numpy()
        num_nodes = x.shape[0]
        a = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
        for i in range(edge_index.shape[1]):
            s, t = edge_index[:, i]
            a[s, t] = 1
            a[t, s] = 1
        a = a.tocsr()
        num_classes = labels.max() + 1
        y_onehot = np.eye(num_classes)[labels]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Prepare PyG data
    row, col = a.nonzero()
    edge_index = np.vstack([row, col])
    pyg = PyGData(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(labels, dtype=torch.long)
    )

    G = nx.from_scipy_sparse_array(a)

    return {
        "x": np.array(x, dtype=float),
        "a": a.tocsr(),
        "y": np.array(y_onehot, dtype=float),
        "labels": np.array(labels, dtype=int),
        "G": G,
        "pyg_data": pyg
    }

# ----------------------------
# Masking utility (allow import from file)
# ----------------------------
def create_label_mask(labels, mask_frac=0.7, seed=None, mask_indices_path=None):
    """
    Create masked labels according to mask_frac (fraction of nodes to KEEP KNOWN).
    If mask_indices_path is given, load array of indices from that path and use it as the masked indices.

    Returns:
      masked_labels: int array where masked positions are -1, known positions contain label ints
      label_mask: boolean array True for KNOWN labels
      labels_to_be_masked: indices of masked nodes (as np.array)
    """
    n = len(labels)
    rng = np.random.RandomState(seed) if seed is not None else np.random
    if mask_indices_path is not None:
        idx = np.load(mask_indices_path)
        labels_to_be_masked = np.array(idx, dtype=int)
    else:
        # Fraction KNOWN = mask_frac → fraction MASKED = 1 - mask_frac
        k = int(round(n * (1 - mask_frac)))
        labels_to_be_masked = rng.choice(np.arange(n), size=k, replace=False)

    masked = np.full(n, -1, dtype=int)
    mask_set = set(labels_to_be_masked.tolist())
    for i in range(n):
        if i not in mask_set:
            masked[i] = labels[i]
    label_mask = masked != -1
    return masked, label_mask, labels_to_be_masked

# ----------------------------
# Embedding generation functions
# ----------------------------
def deepwalk_embedding(G, k=DEFAULT_EMB_DIM, workers=1, p=1, q=1, seed=None):
    """
    Uses Node2Vec class to produce DeepWalk-style embeddings (default parameters).
    """
    node2vec = Node2Vec(G, dimensions=k, workers=workers, p=1, q=1, seed=seed)
    model = node2vec.fit()  # default window etc
    # model.wv keys are strings in your previous code — ensure node IDs are strings
    return np.vstack([model.wv[str(n)] for n in G.nodes()])

def node2vec_embedding(G, k=DEFAULT_EMB_DIM, workers=1, p=0.5, q=2, seed=None):
    node2vec = Node2Vec(G, dimensions=k, workers=workers, p=p, q=q, seed=seed)
    model = node2vec.fit()
    return np.vstack([model.wv[str(n)] for n in G.nodes()])

def random_embedding(n_nodes, k=DEFAULT_EMB_DIM, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random
    return rng.randn(n_nodes, k)

def given_embedding(features):
    """
    If given features differ from required k, project using TruncatedSVD (if larger) or pad with zeros (if smaller).
    """
    X = np.array(features, dtype=float)
    return X

# VGAE & DGI (PyG) — default-ish training loops, one-hot features used as in original notebooks
def vgae_embedding(pyg_data, k=DEFAULT_EMB_DIM, epochs=200, device='cpu'):
    device = torch.device(device)
    num_nodes = pyg_data.num_nodes
    x = torch.eye(num_nodes, device=device)
    edge_index = pyg_data.edge_index.to(device)

    class Encoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = PyG_GCNConv(in_channels, 2 * out_channels)
            self.conv_mu = PyG_GCNConv(2 * out_channels, out_channels)
            self.conv_logstd = PyG_GCNConv(2 * out_channels, out_channels)
        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            mu = self.conv_mu(x, edge_index)
            logstd = self.conv_logstd(x, edge_index)
            return mu, logstd

    model = PyG_VGAE(Encoder(num_nodes, k)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index) + (1.0 / num_nodes) * model.kl_loss()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
    return z.detach().cpu().numpy()

def dgi_embedding(pyg_data, k=150, epochs=200, device='cpu'):
    device = torch.device(device)
    num_nodes = pyg_data.num_nodes
    x = torch.randn(num_nodes, k, device=device)  # Random initialization

    edge_index = pyg_data.edge_index.to(device)

    class GCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = PyG_GCNConv(in_channels, 2 * out_channels)
            self.conv2 = PyG_GCNConv(2 * out_channels, out_channels)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    model = DeepGraphInfomax(
        hidden_channels=k,
        encoder=GCNEncoder(in_channels=k, out_channels=k),
        summary=lambda z, *args, **kwargs: torch.mean(z, dim=0),
        corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pos_z, neg_z, summary = model(x, edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pos_z, _, _ = model(x, edge_index)

    return pos_z.detach().cpu().numpy()

# ----------------------------
# FUSE: modularity-based embedding (your semi_supervised_gradient_ascent_modularity)
# ----------------------------
def perform_labeled_random_walks(G, label_mask, labels, num_walks=10, walk_length=5, walk_length_labelled=3):
    walks = {node: [] for node in G.nodes()}
    nodes = list(G.nodes())
    for node in nodes:
        for _ in range(num_walks):
            walk = [node]
            labeled_count = 0
            for _ in range(walk_length - 1):
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if not neighbors:
                    break
                labeled_neighbors = [n for n in neighbors if label_mask[n]]
                if labeled_neighbors and labeled_count < walk_length_labelled:
                    next_node = random.choice(labeled_neighbors)
                    labeled_count += 1
                else:
                    next_node = random.choice(neighbors)
                walk.append(next_node)
            walks[node].extend([n for n in walk if label_mask[n]])
    return walks

def compute_attention_weights(S, labeled_nodes):
    weights = {}
    for node, labeled in labeled_nodes.items():
        if labeled:
            similarities = {n: float(np.dot(S[node], S[n])) for n in labeled}
            exp_sims = {n: np.exp(sim) for n, sim in similarities.items()}
            total = sum(exp_sims.values()) if len(exp_sims) > 0 else 1.0
            weights[node] = {n: exp_sims[n] / total for n in labeled}
    return weights

def fuse_embedding(G, labels, label_mask, k=DEFAULT_EMB_DIM, eta=0.01, lambda_supervised=1.0,
                   lambda_semi=2.0, iterations=200, initialization='random',
                   num_walks=10, walk_length=5, walk_length_labelled=3, seed=None):
    """
    FUSE: your semi-supervised modularity-based embedding
    Returns S (n_nodes x k) numpy array
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    A = csr_matrix(nx.to_scipy_sparse_array(G, format='csr'))
    degrees = np.array(A.sum(axis=1)).flatten()
    m = G.number_of_edges()
    n = A.shape[0]

    if initialization == 'random':
        S = np.random.randn(n, k)
    else:
        S = np.random.randn(n, k)
    # Orthonormalize columns
    S, _ = np.linalg.qr(S)

    labeled_walks = perform_labeled_random_walks(G, label_mask, labels, num_walks, walk_length, walk_length_labelled)
    attention_weights = compute_attention_weights(S, labeled_walks)

    for _ in range(iterations):
        neighbor_agg = A @ S
        global_correction = (degrees[:, None] / (2 * m)) * S.sum(axis=0)
        grad_modularity = (1 / (2 * m)) * (neighbor_agg - global_correction)

        grad_supervised = np.zeros_like(S)
        if np.any(label_mask):
            unique_labels = np.unique(labels[label_mask])
            for lab in unique_labels:
                mask = (labels == lab) & label_mask
                if mask.sum() == 0:
                    continue
                mean_embedding = np.mean(S[mask], axis=0, keepdims=True)
                grad_supervised[mask] = S[mask] - mean_embedding

        grad_semi_supervised = np.zeros_like(S)
        for i in range(n):
            if (not label_mask[i]) and (i in attention_weights):
                weighted_embedding = sum(w * S[j] for j, w in attention_weights[i].items())
                grad_semi_supervised[i] = S[i] - weighted_embedding

        grad_total = grad_modularity - lambda_supervised * grad_supervised - lambda_semi * grad_semi_supervised
        S += eta * grad_total
        S, _ = np.linalg.qr(S)

    return S

def fuse_embedding_gpu(G, labels, label_mask, k=DEFAULT_EMB_DIM, eta=0.01,
                       lambda_supervised=1.0, lambda_semi=2.0,
                       iterations=200, seed=None,
                       num_walks=10, walk_length=5, walk_length_labelled=3,
                       device='cuda'):
    """
    GPU-accelerated FUSE using PyTorch sparse tensors.
    Falls back to CPU fuse_embedding if CUDA unavailable.
    """
    if not torch.cuda.is_available():
        print("[FUSE] CUDA not available, falling back to CPU.")
        return fuse_embedding(G, labels, label_mask, k=k, eta=eta,
                              lambda_supervised=lambda_supervised,
                              lambda_semi=lambda_semi, iterations=iterations,
                              seed=seed)

    dev = torch.device(device)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    # Build sparse adjacency as torch sparse tensor on GPU
    A_scipy = csr_matrix(nx.to_scipy_sparse_array(G, format='csr')).tocoo()
    indices = torch.tensor(
        np.vstack([A_scipy.row, A_scipy.col]), dtype=torch.long, device=dev
    )
    values = torch.tensor(A_scipy.data, dtype=torch.float32, device=dev)
    n = A_scipy.shape[0]
    A = torch.sparse_coo_tensor(indices, values, (n, n), device=dev)

    degrees = torch.tensor(
        np.array(A_scipy.sum(axis=1)).flatten(), dtype=torch.float32, device=dev
    )
    m = G.number_of_edges()

    # Initialize S on GPU
    S = torch.randn(n, k, device=dev, dtype=torch.float32)
    S, _ = torch.linalg.qr(S)
    S = S.contiguous()

    # Random walks run on CPU (graph traversal, not parallelizable on GPU)
    label_mask_np = np.array(label_mask)
    labels_np = np.array(labels)
    labeled_walks = perform_labeled_random_walks(
        G, label_mask_np, labels_np, num_walks, walk_length, walk_length_labelled
    )

    # Precompute attention weights on CPU, convert to GPU tensors
    S_cpu = S.cpu().numpy()
    attention_weights = compute_attention_weights(S_cpu, labeled_walks)

    label_mask_t = torch.tensor(label_mask_np, dtype=torch.bool, device=dev)
    labels_t = torch.tensor(labels_np, dtype=torch.long, device=dev)
    unique_labels = np.unique(labels_np[label_mask_np])

    for iteration in range(iterations):
        # Sparse matmul on GPU
        neighbor_agg = torch.sparse.mm(A, S)
        global_correction = (degrees.unsqueeze(1) / (2 * m)) * S.sum(dim=0)
        grad_modularity = (1.0 / (2 * m)) * (neighbor_agg - global_correction)

        # Supervised gradient on GPU
        grad_supervised = torch.zeros_like(S)
        for lab in unique_labels:
            mask = (labels_t == int(lab)) & label_mask_t
            if mask.sum() == 0:
                continue
            mean_emb = S[mask].mean(dim=0, keepdim=True)
            grad_supervised[mask] = S[mask] - mean_emb

        # Semi-supervised gradient (attention weights precomputed on CPU)
        grad_semi = torch.zeros_like(S)
        for i, weights in attention_weights.items():
            if label_mask_np[i]:
                continue
            if not weights:
                continue
            weighted = torch.zeros(k, device=dev, dtype=torch.float32)
            for j, w in weights.items():
                weighted += w * S[j]
            grad_semi[i] = S[i] - weighted

        # Gradient step
        S = S + eta * (grad_modularity
                       - lambda_supervised * grad_supervised
                       - lambda_semi * grad_semi)

        # QR on GPU
        S, _ = torch.linalg.qr(S)
        S = S.contiguous()

    torch.cuda.synchronize()
    return S.cpu().numpy()

def fuse_embedding_gpu(G, labels, label_mask, k=DEFAULT_EMB_DIM, eta=0.01,
                       lambda_supervised=1.0, lambda_semi=2.0,
                       iterations=200, seed=None,
                       num_walks=10, walk_length=5, walk_length_labelled=3,
                       device='cuda'):
    if not torch.cuda.is_available():
        print("[FUSE] CUDA not available, falling back to CPU.")
        return fuse_embedding(G, labels, label_mask, k=k, eta=eta,
                              lambda_supervised=lambda_supervised,
                              lambda_semi=lambda_semi, iterations=iterations, seed=seed)

    dev = torch.device(device)
    if seed is not None:
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)

    # --- Adjacency as GPU sparse tensor ---
    A_scipy = csr_matrix(nx.to_scipy_sparse_array(G, format='csr')).tocoo()
    n = A_scipy.shape[0]
    indices = torch.tensor(np.vstack([A_scipy.row, A_scipy.col]), dtype=torch.long, device=dev)
    values  = torch.tensor(A_scipy.data, dtype=torch.float32, device=dev)
    A = torch.sparse_coo_tensor(indices, values, (n, n), device=dev)

    degrees = torch.tensor(
        np.array(csr_matrix(nx.to_scipy_sparse_array(G)).sum(axis=1)).flatten(),
        dtype=torch.float32, device=dev
    )
    m = G.number_of_edges()

    # --- Initialize S ---
    S = torch.randn(n, k, device=dev, dtype=torch.float32)
    S, _ = torch.linalg.qr(S)
    S = S.contiguous()

    # --- Labels on GPU ---
    label_mask_np = np.array(label_mask)
    labels_np     = np.array(labels)
    label_mask_t  = torch.tensor(label_mask_np, dtype=torch.bool, device=dev)
    labels_t      = torch.tensor(labels_np, dtype=torch.long, device=dev)
    unique_labels  = np.unique(labels_np[label_mask_np])
    unlabeled_idx  = np.where(~label_mask_np)[0]

    # --- Build attention weight sparse matrix (CPU walks → GPU sparse tensor) ---
    labeled_walks = perform_labeled_random_walks(
        G, label_mask_np, labels_np, num_walks, walk_length, walk_length_labelled
    )

    # Softmax-normalize the walk co-occurrence counts into a sparse weight matrix
    # Shape: (n, n), rows are unlabeled nodes, cols are the labeled nodes they walked to
    rows, cols, vals = [], [], []
    for i in unlabeled_idx:
        neighbors = labeled_walks.get(i, [])
        if not neighbors:
            continue
        # Count occurrences (matches compute_attention_weights logic)
        from collections import Counter
        counts = Counter(neighbors)
        exp_counts = {j: np.exp(float(c)) for j, c in counts.items()}
        total = sum(exp_counts.values())
        for j, ev in exp_counts.items():
            rows.append(i)
            cols.append(j)
            vals.append(ev / total)

    if rows:
        W_indices = torch.tensor([rows, cols], dtype=torch.long, device=dev)
        W_values  = torch.tensor(vals, dtype=torch.float32, device=dev)
        W = torch.sparse_coo_tensor(W_indices, W_values, (n, n), device=dev).coalesce()
    else:
        W = None  # no semi-supervised signal

    # --- Main training loop (fully vectorized) ---
    for _ in range(iterations):
        # Modularity gradient
        neighbor_agg      = torch.sparse.mm(A, S)
        global_correction = (degrees.unsqueeze(1) / (2 * m)) * S.sum(dim=0)
        grad_modularity   = (1.0 / (2 * m)) * (neighbor_agg - global_correction)

        # Supervised gradient (vectorized over label classes)
        grad_supervised = torch.zeros_like(S)
        for lab in unique_labels:
            mask = (labels_t == int(lab)) & label_mask_t
            if mask.sum() == 0:
                continue
            mean_emb = S[mask].mean(dim=0, keepdim=True)
            grad_supervised[mask] = S[mask] - mean_emb

        # Semi-supervised gradient: one sparse matmul instead of Python loop
        if W is not None:
            weighted_agg = torch.sparse.mm(W, S)   # (n, k) — zero for labeled nodes
            grad_semi    = S - weighted_agg
            grad_semi[label_mask_t] = 0.0           # only apply to unlabeled nodes
        else:
            grad_semi = torch.zeros_like(S)

        # Update + re-orthogonalize
        S = S + eta * (grad_modularity
                       - lambda_supervised * grad_supervised
                       - lambda_semi * grad_semi)
        S, _ = torch.linalg.qr(S)
        S = S.contiguous()

    return S.cpu().numpy()


# ----------------------------
# Classifier models (Spektral-based, preserve structure)
# ----------------------------
class NoMaskGCNConv(GCNConv):
    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, training=None, mask=None):
        # Explicitly discard mask
        return super().call(inputs, mask=None)
        
class GCN(tf.keras.Model):
    def __init__(self, n_labels, seed=42):
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.conv1 = NoMaskGCNConv(16, activation='relu', kernel_initializer=initializer)
        self.conv2 = NoMaskGCNConv(n_labels, activation='softmax', kernel_initializer=initializer)

    def call(self, inputs, training=False):
        x, a = inputs
        intermediate_embeddings = self.conv1([x, a])
        x = self.conv2([intermediate_embeddings, a])
        return x, intermediate_embeddings

# Define a custom wrapper for GATConv that avoids mask issues
class NoMaskGATConv(GATConv):
    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, training=None, mask=None):
        # Explicitly discard the mask argument
        return super().call(inputs, mask=None)

# Define the GAT model using the NoMaskGATConv
class GAT(tf.keras.Model):
    def __init__(self, n_labels, num_heads=8, seed=42):
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        # Use the custom NoMaskGATConv instead of the original GATConv
        self.conv1 = NoMaskGATConv(16, attn_heads=num_heads, concat_heads=True, activation='elu', kernel_initializer=initializer)
        self.conv2 = NoMaskGATConv(n_labels, attn_heads=1, concat_heads=False, activation='softmax', kernel_initializer=initializer)

    def call(self, inputs):
        x, a = inputs
        intermediate_embeddings = self.conv1([x, a])  # Store intermediate embeddings
        x = self.conv2([intermediate_embeddings, a])
        return x, intermediate_embeddings  # Return both final output and intermediate embeddings

class GraphSAGE(tf.keras.Model):
    def __init__(self, n_labels, hidden_dim=16, aggregator='mean', seed=42):
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.conv1 = GraphSageConv(hidden_dim, activation='relu', aggregator=aggregator, kernel_initializer=initializer)
        self.conv2 = GraphSageConv(n_labels, activation='softmax', aggregator=aggregator, kernel_initializer=initializer)
    def call(self, inputs, training=False):
        x, a = inputs
        interm = self.conv1([x, a])
        out = self.conv2([interm, a])
        return out, interm

# ----------------------------
# Training / evaluation helpers
# ----------------------------

def sparse_to_tf_sparse(A):
    """Convert scipy csr_matrix to tf.sparse.SparseTensor"""
    A = A.tocoo()
    indices = np.column_stack((A.row, A.col))
    return tf.sparse.SparseTensor(
        indices=indices,
        values=A.data,
        dense_shape=A.shape
    )

def evaluate_preds(true_int_labels, pred_int_labels):
    acc = accuracy_score(true_int_labels, pred_int_labels)
    f1 = f1_score(true_int_labels, pred_int_labels, average='macro')
    cm = confusion_matrix(true_int_labels, pred_int_labels)
    return {"accuracy": float(acc), "f1_score": float(f1), "confusion_matrix": cm}

def train_and_evaluate_classifier(embedding_matrix, adjacency_csr, labels_onehot, labels_int, label_mask,
                                  classifier_name='gcn', epochs=200, seed=42, verbose=False):
    """
    Train the chosen classifier using embedding_matrix as features and adjacency.
    label_mask: boolean array True for nodes whose labels are KNOWN (used in training)
    Returns:
      results dict with accuracy/f1 on masked nodes, training_time_seconds, predictions for all nodes (ints)
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n_nodes = embedding_matrix.shape[0]
    num_classes = labels_onehot.shape[1]
    
    # Get training indices (nodes with known labels)
    train_idx = np.where(label_mask)[0]
    
    # Create training subgraph
    X_train = tf.convert_to_tensor(np.array(embedding_matrix[train_idx], dtype=np.float32))
    y_train = labels_onehot[train_idx]
    
    # Reduce the adjacency matrix to only include training nodes
    A_train = adjacency_csr[train_idx, :][:, train_idx]
    A_train_sp = sparse_to_tf_sparse(A_train)

    # instantiate model
    if classifier_name.lower() == 'gcn':
        model = GCN(num_classes, seed=seed)
    elif classifier_name.lower() == 'gat':
        model = GAT(num_classes, seed=seed)
    elif classifier_name.lower() in ('graphsage', 'sage', 'graph_sage'):
        model = GraphSAGE(num_classes, seed=seed)
    else:
        raise ValueError("Unknown classifier: " + classifier_name)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    num_train = len(train_idx)
    num_test = len(labels_int) - num_train
    if verbose:
        print(f"[Classifier={classifier_name}] Train={num_train}, Test={num_test}")

    t0 = time.time()
    # training loop
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            preds, _ = model([X_train, A_train_sp], training=True)  # (n_train_nodes, num_classes)
            loss = loss_fn(y_train, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # minimal printing
        if verbose and (epoch % 50 == 0):
            print(f"[{classifier_name}] epoch {epoch}, loss={float(loss):.4f}")
    training_time = time.time() - t0

    # Prepare the full graph for prediction
    X_full = tf.convert_to_tensor(np.array(embedding_matrix, dtype=np.float32))
    A_full_sp = sparse_to_tf_sparse(adjacency_csr)
    
    # Make predictions for all nodes
    preds_all, emb_intermediate = model([X_full, A_full_sp], training=False)
    pred_int = tf.argmax(preds_all, axis=1).numpy()

    # masked nodes = those NOT in train_idx
    masked_idx = np.where(~label_mask)[0]
    true_masked = labels_int[masked_idx]
    pred_masked = pred_int[masked_idx]
    results = evaluate_preds(true_masked, pred_masked)
    results.update({
        "training_time_seconds": float(training_time),
        "classifier": classifier_name,
        "predictions_all": pred_int,
        "masked_indices": masked_idx
    })
    return results

# ----------------------------
# Orchestrator: run single dataset + seed + mask_frac
# ----------------------------
def run_one_experiment(dataset_name, seed, mask_frac, emb_dim=DEFAULT_EMB_DIM,
                       embedding_methods=None, classifiers=None,
                       mask_indices_path=None, vgae_epochs=200, dgi_epochs=200,
                       fuse_iterations=200, force_device='cpu',
                       save_dir="./benchmark_outputs", verbose=False):
    """
    Run all embeddings and classifiers for one dataset/seed/mask_frac.
    Prints progress if verbose=True.
    Saves embeddings in folder structure:
        <save_dir>/<dataset>/<masked%-known%>/
        Filenames: <embedding>_embedding_<masked%>_<known%>_<seed>.pkl
    """
    import pickle

    if embedding_methods is None:
        embedding_methods = ['random', 'given', 'deepwalk', 'node2vec', 'vgae', 'dgi', 'fuse']
    if classifiers is None:
        classifiers = ['gcn', 'gat', 'graphsage']

    ds = load_dataset(dataset_name)
    X = ds['x']
    A = ds['a']
    y_onehot = ds['y']
    labels_int = ds['labels']
    G = ds['G']
    pyg = ds['pyg_data']
    n = X.shape[0]

    # Build masks
    masked_labels, label_mask, labels_to_be_masked = create_label_mask(
        labels_int, mask_frac, seed=seed, mask_indices_path=mask_indices_path
    )

    num_masked = len(labels_to_be_masked)
    num_unmasked = n - num_masked
    if verbose:
        print(f"[{dataset_name}][seed={seed}][mf={mask_frac}] "
              f"Masked={num_masked}, Unmasked={num_unmasked}")

    embedding_times = []
    embeddings = {}
    run_results = []

    masked_pct = int(mask_frac * 100)
    known_pct = 100 - masked_pct
    folder_name = f"{masked_pct}-{known_pct}"  # e.g., "70-30"
    dsdir = os.path.join(save_dir, dataset_name, folder_name)
    os.makedirs(dsdir, exist_ok=True)

    for emb_name in embedding_methods:
        if verbose:
            print(f"[{dataset_name}][seed={seed}][mask_frac={mask_frac}] Running {emb_name} …")

        tstart = time.time()
        # Generate embedding
        if emb_name.lower() == 'random':
            E = random_embedding(n, k=emb_dim, seed=seed)
        elif emb_name.lower() == 'given':
            E = given_embedding(X)
        elif emb_name.lower() == 'deepwalk':
            E = deepwalk_embedding(G, k=emb_dim, seed=seed)
        elif emb_name.lower() == 'node2vec':
            E = node2vec_embedding(G, k=emb_dim, seed=seed)
        # Around line where embeddings are generated, add cache clearing
        elif emb_name.lower() == 'vgae':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            E = vgae_embedding(pyg, k=emb_dim, epochs=vgae_epochs, device=force_device)
        elif emb_name.lower() == 'dgi':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            E = dgi_embedding(pyg, k=emb_dim, epochs=dgi_epochs, device=force_device)
        elif emb_name.lower() in ('fuse', 'modularity'):
            if force_device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()
                E = fuse_embedding_gpu(G, labels_int, label_mask, k=emb_dim,
                                       iterations=fuse_iterations, seed=seed,
                                       device=force_device)
            else:
                E = fuse_embedding(G, labels_int, label_mask, k=emb_dim,
                                   iterations=fuse_iterations, seed=seed)
        else:
            raise ValueError("Unknown embedding name: " + emb_name)

        t_elapsed = time.time() - tstart
        embedding_times.append((emb_name, t_elapsed))
        embeddings[emb_name] = E

        # Save embedding
        filename = f"{emb_name.lower()}_embedding_{masked_pct}_{known_pct}_{seed}.pkl"
        with open(os.path.join(dsdir, filename), "wb") as f:
            pickle.dump(E, f)

        # Run classifiers
        for clf in classifiers:
            clf_start = time.time()
            res = train_and_evaluate_classifier(E, A, y_onehot, labels_int, label_mask,
                                                classifier_name=clf, epochs=200, seed=seed, verbose=False)
            clf_time = time.time() - clf_start
            
            # Print results for each case
            print(f"Embedding: {emb_name}, Classifier: {clf}")
            print(f"Accuracy: {res['accuracy']:.4f}")
            print(f"Embedding generation time: {t_elapsed:.2f}s")
            print(f"Classifier runtime: {clf_time:.2f}s")
            print("-" * 50)
            
            res_meta = {
                "dataset": dataset_name,
                "seed": seed,
                "mask_frac": mask_frac,
                "embedding": emb_name,
                "classifier": clf,
                "embedding_time_seconds": float(t_elapsed),
                "train_time_seconds": float(res.get("training_time_seconds", np.nan)),
                "accuracy": float(res["accuracy"]),
                "f1_score": float(res["f1_score"])
            }
            run_results.append(res_meta)

    return run_results, embedding_times, embeddings

# ----------------------------
# Helper to build mask file path
# ----------------------------
def get_mask_file_path(masks_root, dataset_name, seed, mask_frac):
    """
    Build expected mask file path given dataset, seed, mask_frac (fraction masked).
    Returns None if no such file exists.
    
    Folder structure:
      masks_root/
        AmazonPhotos/
          70_30/   <- 70% masked, 30% known
          30_70/   <- 30% masked, 70% known
        Cora/...
        PubMed/...
        WikiCS/...
        CiteSeer/...
    """
    dataset_map = {
        "cora": "Cora",
        "citeseer": "CiteSeer",
        "pubmed": "PubMed",
        "wikics": "WikiCS",
        "photo": "AmazonPhotos",
        "amazon-photo": "AmazonPhotos",
        "amazon_photos": "AmazonPhotos"
    }
    if dataset_name.lower() not in dataset_map:
        return None  # no mapping -> skip
    
    folder_name = dataset_map[dataset_name.lower()]
    if mask_frac == 0.7:
        subfolder = "70_30"  # 70% masked, 30% known
        fname = f"{folder_name}_70_30_masked_indices_seed{seed}.npy"
    elif mask_frac == 0.3:
        subfolder = "30_70"  # 30% masked, 70% known
        fname = f"{folder_name}_30_70_masked_indices_seed{seed}.npy"
    else:
        return None
    
    mask_path = os.path.join(masks_root, folder_name, subfolder, fname)
    return mask_path if os.path.exists(mask_path) else None

# ----------------------------
# High-level run_benchmark driver
# ----------------------------
def run_benchmark(datasets, seeds, mask_fracs=[0.7, 0.3], emb_dim=DEFAULT_EMB_DIM,
                  embedding_methods=None, classifiers=None,
                  vgae_epochs=200, dgi_epochs=200, fuse_iterations=200,
                  save_dir="./benchmark_outputs", device='cpu',
                  masks_root="./masks", verbose=False):
    """
    Run full benchmark across datasets, seeds, mask fractions.
    Saves results per dataset × split in folders like:
        <save_dir>/<dataset>/<masked%-known%>/per_run_results.csv
    Final merged CSVs also saved in <save_dir>
    """
    import pandas as pd
    all_results = []
    all_embedding_times = []

    tasks = [(ds, mf) for ds in datasets for mf in mask_fracs]

    for ds, mf in tqdm(tasks, desc="Benchmark tasks", disable=verbose):
        partial_results = []
        partial_times = []
        for seed in seeds:
            mask_path = get_mask_file_path(masks_root, ds, seed, mf)
            if verbose:
                if mask_path:
                    print(f"Using custom mask: {mask_path}")
                else:
                    print(f"No custom mask for [{ds} seed={seed} mf={mf}] → random mask.")

            rr, et, _ = run_one_experiment(
                ds, seed, mf, emb_dim=emb_dim,
                embedding_methods=embedding_methods,
                classifiers=classifiers,
                vgae_epochs=vgae_epochs,
                dgi_epochs=dgi_epochs,
                fuse_iterations=fuse_iterations,
                force_device=device,
                save_dir=save_dir,
                verbose=verbose,
                mask_indices_path=mask_path
            )
            partial_results.extend(rr)
            for e in et:
                e_rec = {"dataset": ds, "seed": seed, "mask_frac": mf,
                         "embedding": e[0], "time_seconds": float(e[1])}
                partial_times.append(e_rec)

        # Save per dataset × split
        masked_pct = int(mf * 100)
        known_pct = 100 - masked_pct
        dsdir = os.path.join(save_dir, ds, f"{masked_pct}-{known_pct}")
        os.makedirs(dsdir, exist_ok=True)

        pd.DataFrame(partial_results).to_csv(os.path.join(dsdir, "per_run_results.csv"), index=False)
        pd.DataFrame(partial_times).to_csv(os.path.join(dsdir, "embedding_times.csv"), index=False)

        all_results.extend(partial_results)
        all_embedding_times.extend(partial_times)

    # Final aggregation
    df_results = pd.DataFrame(all_results)
    df_embtimes = pd.DataFrame(all_embedding_times)

    avg_by_model = df_results.groupby(
        ["dataset", "mask_frac", "embedding", "classifier"]
    ).agg(
        avg_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
        avg_f1=("f1_score", "mean"),
        std_f1=("f1_score", "std"),
        avg_train_time=("train_time_seconds", "mean"),
        std_train_time=("train_time_seconds", "std"),
        n_runs=("accuracy", "count")
    ).reset_index()

    avg_by_model["accuracy_pm"] = avg_by_model.apply(
        lambda r: f"{r['avg_accuracy']:.4f} ± {r['std_accuracy']:.4f}", axis=1
    )
    avg_by_model["f1_pm"] = avg_by_model.apply(
        lambda r: f"{r['avg_f1']:.4f} ± {r['std_f1']:.4f}", axis=1
    )

    avg_embtime = df_embtimes.groupby(
        ["dataset", "mask_frac", "embedding"]
    ).agg(
        avg_embedding_time=("time_seconds", "mean"),
        std_embedding_time=("time_seconds", "std"),
        n_runs=("time_seconds", "count")
    ).reset_index()

    avg_embtime["embedding_time_pm"] = avg_embtime.apply(
        lambda r: f"{r['avg_embedding_time']:.4f} ± {r['std_embedding_time']:.4f}", axis=1
    )

    os.makedirs(save_dir, exist_ok=True)
    df_results.to_csv(os.path.join(save_dir, "per_run_results_all.csv"), index=False)
    avg_by_model.to_csv(os.path.join(save_dir, "avg_by_model_and_classifier.csv"), index=False)
    avg_embtime.to_csv(os.path.join(save_dir, "avg_embedding_times.csv"), index=False)

    return {
        "per_run": df_results,
        "avg_by_model_and_classifier": avg_by_model,
        "avg_embedding_times": avg_embtime
    }
