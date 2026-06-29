# import os
# import random
# import time 
import networkx as nx
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import umap.umap_ as umap
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from tqdm import tqdm
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import CategoricalCrossentropy
# from tensorflow.keras.metrics import CategoricalAccuracy
import torch
# from torch_geometric.data import Data
# import spektral
# from spektral.layers import GCNConv, GATConv
# from spektral.layers import GraphSageConv
# from spektral.data import Graph, Dataset, BatchLoader
from scipy.sparse import csr_matrix, lil_matrix
# from torch_geometric.datasets import Amazon
# from torch_geometric.utils import from_networkx
# import scipy.sparse as sp
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# from scipy.sparse.csgraph import laplacian
# from scipy.sparse.linalg import eigsh
# from collections import Counter
# from sklearn.preprocessing import normalize
# from joblib import Parallel, delayed
# from torch_geometric.data import Data

# Unsupervised gradient ascent for modularity maximization
def gradient_ascent_modularity_unsupervised(G, k=2, eta=0.01, iterations=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    A = nx.to_numpy_array(G)
    l = A.sum(axis=1)
    m = np.sum(l) / 2
    B = A - np.outer(l, l) / (2 * m)
    n = B.shape[0]

    S = rng.randn(n, k)  # Random Initialization
    S, _ = np.linalg.qr(S)  # Ensure initial orthonormality

    for _ in tqdm(range(iterations), desc="Gradient Ascent Progress"):
        grad = (1 / (2 * m)) * B @ S
        S += eta * grad
        S, _ = np.linalg.qr(S)  # Orthonormalize using QR decomposition

    return S

def perform_labeled_random_walks(G, label_mask, labels, num_walks, walk_length, walk_length_labelled=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    walks = {node: [] for node in G.nodes()}
    for node in G.nodes():
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
                    next_node = rng.choice(labeled_neighbors)
                    labeled_count += 1
                else:
                    next_node = rng.choice(neighbors)
                walk.append(next_node)
            walks[node].extend([n for n in walk if label_mask[n]])
    return walks

def compute_attention_weights(S, labeled_nodes):
    weights = {}
    for node, labeled in labeled_nodes.items():
        if labeled:
            similarities = {n: np.dot(S[node], S[n]) for n in labeled}
            exp_sims = {n: np.exp(sim) for n, sim in similarities.items()}
            total = sum(exp_sims.values())
            weights[node] = {n: exp_sims[n] / total for n in labeled}
    return weights

def semi_supervised_gradient_ascent_modularity(G, labels, label_mask, k=2, eta=0.01, lambda_supervised=1.0, 
                                                      lambda_semi=2.0, iterations=5000, initialization='random',
                                                      num_walks=10, walk_length=5, walk_length_labelled=3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Convert graph to sparse adjacency matrix
    A = csr_matrix(nx.to_scipy_sparse_array(G, format='csr'))
    degrees = np.array(A.sum(axis=1)).flatten()
    m = G.number_of_edges()
    n = A.shape[0]

    # Initialize embeddings
    if initialization == 'random':
        if rng is None:
            rng = np.random.default_rng()
        S = rng.normal(size=(n, k))
    S, _ = np.linalg.qr(S)

    # Compute labeled random walks and attention weights
    labeled_walks = perform_labeled_random_walks(G, label_mask, labels, num_walks, walk_length, walk_length_labelled)
    attention_weights = compute_attention_weights(S, labeled_walks)

    for _ in tqdm(range(iterations), desc="Gradient Ascent with Linear Modularity"):
        # Compute modularity gradient using linear approximation
        neighbor_agg = A @ S  # Efficient aggregation of neighbor embeddings
        global_correction = (degrees[:, None] / (2 * m)) * S.sum(axis=0)
        grad_modularity = (1 / (2 * m)) * (neighbor_agg - global_correction)

        # Compute supervised gradient
        grad_supervised = np.zeros_like(S)
        unique_labels = np.unique(labels[label_mask])
        for label in unique_labels:
            mask = (labels == label) & label_mask
            mean_embedding = np.mean(S[mask], axis=0, keepdims=True)
            grad_supervised[mask] = S[mask] - mean_embedding

        # Compute semi-supervised gradient using adaptive attention
        grad_semi_supervised = np.zeros_like(S)
        for i in range(n):
            if not label_mask[i] and i in attention_weights:
                weighted_embedding = sum(weight * S[n] for n, weight in attention_weights[i].items())
                grad_semi_supervised[i] = S[i] - weighted_embedding

        # Update embeddings
        grad_total = grad_modularity - lambda_supervised * grad_supervised - lambda_semi * grad_semi_supervised
        S += eta * grad_total
        S, _ = np.linalg.qr(S)

    return S