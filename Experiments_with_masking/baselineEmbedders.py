# import os
# import random
import time
import networkx as nx
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import umap.umap_ as umap
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from node2vec import Node2Vec
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datasets import AmazonPhotosDataset, CiteSeerDataset, CoraDataset, PubMedDataset, WikiCSDataset, formatDataMissing, convert_to_networkx
from modularityModel import gradient_ascent_modularity_unsupervised, perform_labeled_random_walks, compute_attention_weights, semi_supervised_gradient_ascent_modularity
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
# from scipy.sparse import csr_matrix, lil_matrix
# from torch_geometric.datasets import Amazon
from torch_geometric.nn import DeepGraphInfomax, VGAE
import torch.nn.functional as F
# from torch_geometric.utils import from_networkx
# import scipy.sparse as sp
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
# from scipy.sparse.csgraph import laplacian
# from scipy.sparse.linalg import eigsh
# from collections import Counter
# from sklearn.preprocessing import normalize
# from joblib import Parallel, delayed
from torch_geometric.nn import GCNConv as PyG_GCNConv, VGAE as PyG_VGAE
# from dataclasses import dataclass
from torch_geometric.data import Data
import yaml
from pathlib import Path
from joblib import Memory

here = Path(__file__).resolve().parent
memory = Memory(location=here / 'checkpoints/cache', compress=True)

with open('config.yml', 'r') as file:
    SEED = yaml.safe_load(file)['SEED']

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Laplacian Eigenmaps Embedding
def deepwalk_embedding(
        G,
        k=2,
        walk_length=10,
        num_walks=80,
        workers=4,
        seed=SEED
    ):
    node2vec = Node2Vec(G, dimensions=k, walk_length=walk_length, num_walks=num_walks, workers=workers, seed=seed)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return np.array([model.wv[str(node)] for node in G.nodes()])

# Node2Vec Embedding
def node2vec_embedding(
        G,
        k=2,
        walk_length=10,
        num_walks=100,
        workers=4,
        seed=SEED
    ):
    node2vec = Node2Vec(
        G,
        dimensions=k,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        seed=seed
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    return np.array([model.wv[str(node)] for node in G.nodes()])


# VGAE Embedding 
class VGAEEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = PyG_GCNConv(in_channels, 2 * out_channels)  # Use PyG_GCNConv
        self.conv_mu = PyG_GCNConv(2 * out_channels, out_channels)  # Separate layer for mu
        self.conv_logstd = PyG_GCNConv(2 * out_channels, out_channels)  # Separate layer for logstd

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logstd = self.conv_logstd(x, edge_index)
        return mu, logstd

def vgae_embedding(data, k=128):
    # Use one-hot encoded node IDs as features
    num_nodes = data.num_nodes
    x = torch.eye(num_nodes)  # One-hot encoded node features

    in_channels = x.shape[1]  # Feature dimension is equal to the number of nodes
    model = PyG_VGAE(VGAEEncoder(in_channels, k))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in tqdm(range(200)):
        optimizer.zero_grad()
        z = model.encode(x, data.edge_index)  # Use one-hot encoded features
        loss = model.recon_loss(z, data.edge_index) + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
    
    return model.encode(x, data.edge_index).detach().numpy()

# DGI Embedding
def dgi_embedding(data, k=128):
    class GCNEncoder(torch.nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = PyG_GCNConv(in_channels, 2 * out_channels)  # Use PyG_GCNConv
            self.conv2 = PyG_GCNConv(2 * out_channels, out_channels)  # Use PyG_GCNConv

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            return self.conv2(x, edge_index)

    # Use one-hot encoded node IDs as features
    num_nodes = data.num_nodes
    x = torch.eye(num_nodes)  # One-hot encoded node features

    in_channels = x.shape[1]  # Feature dimension is equal to the number of nodes
    model = DeepGraphInfomax(
        hidden_channels=k,
        encoder=GCNEncoder(in_channels, k),
        summary=lambda z, *args, **kwargs: z.mean(dim=0),  # Ensure `summary` only takes `z`
        corruption=lambda x, edge_index: (x[torch.randperm(x.size(0))], edge_index)  # Correct corruption function
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in tqdm(range(200)):
        optimizer.zero_grad()
        pos_z, neg_z, summary = model(x, data.edge_index)  # Use one-hot encoded features
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()

    return pos_z.detach().numpy()

def drop_edges(edge_index, drop_prob=0.2):
    """Random edge dropout"""
    mask = torch.rand(edge_index.size(1)) > drop_prob
    return edge_index[:, mask]

def drop_features(x, drop_prob=0.2):
    """Random feature dropout"""
    mask = (torch.rand_like(x) > drop_prob).float()
    return x * mask

class SGCL_Loss:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def contrastive_loss(self, z1, z2, tau=0.5):
        # Normalize
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        sim_matrix = torch.mm(z1, z2.t()) / tau
        pos_sim = torch.diag(sim_matrix)
        loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim_matrix).sum(1))
        return loss.mean()

    def smoothing_loss(self, h, L):
        """Graph smoothness regularizer: h^T L h"""
        return torch.trace(h.t() @ L @ h) / h.size(0)

    def compute(self, h1, h2, L):
        con_loss = self.contrastive_loss(h1, h2)
        smooth_loss = (self.smoothing_loss(h1, L) + self.smoothing_loss(h2, L)) / 2
        return con_loss + self.alpha * smooth_loss

def sgcl_embedding(data, k=150, epochs=200, lr=0.001, alpha=0.1, drop_prob=0.2):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    num_nodes = data.num_nodes
    x = torch.randn(num_nodes, k).to(device)   # random init features
    edge_index = data.edge_index.to(device)
    in_dim = x.size(1)

    encoder = torch.nn.Linear(in_dim, k).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loss_fn = SGCL_Loss(alpha=alpha)

    A = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)).to(device),
                                (num_nodes, num_nodes)).to_dense()
    D = torch.diag(A.sum(1))
    L = (D - A).to(device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Augmented view 1
        x1 = drop_features(x, drop_prob)
        e1 = drop_edges(edge_index, drop_prob)
        h1 = encoder(x1)

        # Augmented view 2
        x2 = drop_features(x, drop_prob)
        e2 = drop_edges(edge_index, drop_prob)
        h2 = encoder(x2)

        # Loss
        loss = loss_fn.compute(h1, h2, L)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    return encoder(x).detach().cpu().numpy()

class GraFNEncoder(tf.keras.Model):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(out_dim)

    def call(self, x, training=False):
        h = self.dense1(x)
        if training:
            h = self.dropout(h, training=True)
        z = self.dense2(h)
        return z, h  # z: logits, h: embeddings

class GraFN(tf.keras.Model):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.encoder = GraFNEncoder(hidden_dim, out_dim)

    def call(self, x, training=False):
        _, emb = self.encoder(x, training=training)
        return emb

def drop_features_tf(x, drop_prob=0.2):
    """Random feature dropout"""
    mask = tf.cast(tf.random.uniform(tf.shape(x)) > drop_prob, tf.float32)
    return x * mask

def drop_edges_tf(edge_index, drop_prob=0.2):
    """Random edge dropout (for TensorFlow tensors)"""
    num_edges = tf.shape(edge_index)[1]
    mask = tf.cast(tf.random.uniform((num_edges,)) > drop_prob, tf.bool)
    edge_index_dropped = tf.boolean_mask(edge_index, mask, axis=1)
    return edge_index_dropped

def grafn_embedding(x, num_nodes, edge_index=None, feature_dim=128, hidden_dim=128, out_dim=128, epochs=200, lr=0.01, drop_feat_prob=0.2, drop_edge_prob=0.2):
    """
    x: tf.Tensor or np.array of shape (num_nodes, feature_dim)
    edge_index: optional tf.Tensor of shape (2, num_edges) for edge dropout
    """
    x = tf.random.normal((num_nodes, feature_dim))
    
    model = GraFN(hidden_dim, out_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # --- Augmented views ---
            x1 = drop_features_tf(x, drop_feat_prob)
            x2 = drop_features_tf(x, drop_feat_prob)

            if edge_index is not None:
                e1 = drop_edges_tf(edge_index, drop_edge_prob)
                e2 = drop_edges_tf(edge_index, drop_edge_prob)

            emb1 = model(x1, training=True)
            emb2 = model(x2, training=True)

            # --- Unsupervised consistency loss ---
            emb1 = tf.math.l2_normalize(emb1, axis=1)
            emb2 = tf.math.l2_normalize(emb2, axis=1)
            loss = 2 - 2 * tf.reduce_mean(tf.reduce_sum(emb1 * emb2, axis=1))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, loss: {loss.numpy():.4f}")

    embeddings = model(x, training=False).numpy()
    return embeddings

def record_time(model_name, func, *args, **kwargs):
    print(f"Computing {model_name} embedding...")
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{model_name} embedding computed in {elapsed_time:.2f} seconds.")
    return result, elapsed_time

def to_tf_tensor(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return tf.convert_to_tensor(x, dtype=tf.float32)

@memory.cache
def generate_embeddings(dataset, modelName, hyperparams: dict, rate: float = 0.7, mech: str = 'MCAR', seed=None):

    rng = np.random.default_rng(seed)
    if dataset == 'Cora':
        dataset = CoraDataset()
    elif dataset == 'CiteSeer':
        dataset = CiteSeerDataset()
    elif dataset == 'PubMed':
        dataset = PubMedDataset()
    elif dataset == 'WikiCS':
        dataset = WikiCSDataset()
    elif dataset == 'AmazonPhotos':
        dataset = AmazonPhotosDataset()
    else:
        raise ValueError(f"Dataset {dataset} not found")
    
    X_py, G, A, labels, label_mask, masked_labels, ground_truth_labels, labels_to_be_masked = formatDataMissing(dataset[0], rate, mech, rng)

    if modelName == 'Modularity':
        X_model, elapsed_time = record_time(
            modelName,
            semi_supervised_gradient_ascent_modularity,
            G,
            labels,
            label_mask,
            k=hyperparams['embedding_dim'],
            eta=hyperparams['eta'],
            lambda_supervised=hyperparams['lambda_supervised'],
            lambda_semi=hyperparams['lambda_semi'],
            iterations=hyperparams['iterations'],
            initialization=hyperparams['initialization'],
            num_walks=hyperparams['num_walks'],
            walk_length=hyperparams['walk_length'],
            walk_length_labelled=hyperparams['walk_length_labelled'],
            rng=rng
        )

    elif modelName == 'DeepWalk':
        X_model, elapsed_time = record_time(
            modelName,
            deepwalk_embedding,
            G,
            k=hyperparams['embedding_dim'],
            walk_length=hyperparams['walk_length'],
            num_walks=hyperparams['num_walks'],
            workers=hyperparams['workers'],
            seed=seed
        )
        X_model = tf.convert_to_tensor(X_model, dtype=tf.float32)

    elif modelName == 'VGAE':
        X_model, elapsed_time = record_time(
            modelName,
            vgae_embedding,
            X_py,
            k=hyperparams['embedding_dim']
        )
        
    elif modelName == 'DGI':
        X_model, elapsed_time = record_time(
            modelName,
            dgi_embedding,
            X_py,
            k=hyperparams['embedding_dim']
        )

    elif modelName == 'Node2Vec':
        X_model, elapsed_time = record_time(
            modelName,
            node2vec_embedding,
            G,
            k=hyperparams['embedding_dim'],
            walk_length=hyperparams['walk_length'],
            num_walks=hyperparams['num_walks'],
            workers=hyperparams['workers'],
            seed=seed
        )
    elif modelName == 'GraFN':
        # X_model, elapsed_time = record_time(
        #     modelName,
        #     GraFN_embedding,
        #     X_py,
        #     k=hyperparams['embedding_dim']
        # )
        num_nodes = X_py.num_nodes
        X_model, elapsed_time = record_time(
            "GraFN",
            grafn_embedding,
            X_py,
            num_nodes=num_nodes,
            feature_dim=hyperparams['embedding_dim'],
            hidden_dim=hyperparams['embedding_dim'],
            out_dim=hyperparams['embedding_dim'],
            epochs=hyperparams['epochs'])
        X_model = to_tf_tensor(X_model)
        
    elif modelName == 'SGCL':
        X_model, elapsed_time = record_time("SGCL", sgcl_embedding, X_py, k=hyperparams['embedding_dim'])
        X_model = to_tf_tensor(X_model)

    elif modelName == 'Random':
        print("Generating Random embedding...")
        start_time = time.time()
        shape = (len(ground_truth_labels), hyperparams['embedding_dim'])
        X_model = np.random.randn(*shape)
        X_model = tf.convert_to_tensor(X_model, dtype=tf.float32)
        end_time = time.time()
        print(f"Random embedding generated in {end_time - start_time:.2f} seconds.")
        elapsed_time = end_time - start_time
    else:
        raise ValueError(f"Model {modelName} not found")
    
    return X_model, A, elapsed_time, masked_labels, ground_truth_labels, labels_to_be_masked