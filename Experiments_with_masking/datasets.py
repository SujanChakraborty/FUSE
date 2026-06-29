import networkx as nx
import numpy as np
from spektral.data import Graph, Dataset
from spektral.datasets import Cora
from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import expit
from scipy.optimize import root_scalar
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid, WikiCS
from pathlib import Path

class AmazonPhotosDataset(Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self, path = Path(__file__).resolve().parent / "datasets"):
        dataset = Amazon(root=str(path), name="photo")  # Load Amazon Computers dataset
        graphs = []
        for data in dataset:
            data = dataset

            x = data.x.numpy()
            edge_index = data.edge_index.numpy()
            y = data.y.numpy()

            # One-hot encode labels
            num_classes = y.max() + 1
            y_one_hot = np.eye(num_classes)[y]

            # Convert edge_index to adjacency matrix
            num_nodes = x.shape[0]
            adj = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                adj[src, dst] = 1
                adj[dst, src] = 1
            
            graphs.append(Graph(x=x, a=adj, y=y_one_hot))

        return graphs
    
# Create a custom Dataset for the graph
class CiteSeerDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self, path=Path(__file__).resolve().parent / "datasets"):
        dataset = Planetoid(root=str(path), name="CiteSeer")  # Load CiteSeer dataset
        graphs = []

        for data in dataset:
            # Convert Torch tensors to NumPy
            x = data.x.numpy()
            edge_index = data.edge_index.numpy()
            y = data.y.numpy()

            # One-hot encode labels
            num_classes = y.max() + 1  # Number of classes
            y_one_hot = np.eye(num_classes)[y]  # One-hot encoding

            # Convert edge_index to a sparse adjacency matrix
            num_nodes = x.shape[0]
            adj = csr_matrix((num_nodes, num_nodes))  # Initialize sparse matrix
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                adj[src, dst] = 1
                adj[dst, src] = 1  # Ensure undirected graph

            graphs.append(Graph(x=x, a=adj, y=y_one_hot))

        return graphs
    
# Create a custom Dataset for the graph
class CoraDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        data = Cora()  # Load the dataset
        graphs = [Graph(x=graph.x, a=graph.a, y=graph.y) for graph in data.graphs]
        return graphs
    
# Create a custom Dataset for the graph
class PubMedDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self, path = Path(__file__).resolve().parent / "datasets"):
        dataset = Planetoid(root=str(path), name="PubMed")  # Load CiteSeer dataset
        graphs = []
        
        for data in dataset:
            # Convert Torch tensors to NumPy
            x = data.x.numpy()
            edge_index = data.edge_index.numpy()
            y = data.y.numpy()

            # One-hot encode labels
            num_classes = y.max() + 1  # Number of classes
            y_one_hot = np.eye(num_classes)[y]  # One-hot encoding
            # Convert edge_index to a sparse adjacency matrix
            num_nodes = x.shape[0]
            adj = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                adj[src, dst] = 1
                adj[dst, src] = 1  # Ensure undirected graph
            graphs.append(Graph(x=x, a=adj, y=y_one_hot))
        return graphs


import numpy as np
# Create a custom Dataset
class WikiCSDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self, path = Path(__file__).resolve().parent / "datasets"):
        dataset = WikiCS(root=str(path / "WikiCS"))  # Load WikiCS dataset
        graphs = []
        
        for data in dataset:
            # Convert to NumPy
            x = data.x.numpy()
            edge_index = data.edge_index.numpy()
            y = data.y.numpy()

            # One-hot encode labels
            num_classes = y.max() + 1
            y_one_hot = np.eye(num_classes)[y]

            # Convert edge_index to sparse adjacency matrix
            num_nodes = x.shape[0]
            adj = lil_matrix((num_nodes, num_nodes), dtype=np.float32)
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                adj[src, dst] = 1
                adj[dst, src] = 1  # Ensure undirected graph
            graphs.append(Graph(x=x, a=adj, y=y_one_hot))

        return graphs

def downloadDatasets(datasets, path = Path(__file__).resolve().parent / "datasets"):

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
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
    
def convert_to_networkx(A):
    return nx.from_scipy_sparse_array(A)

def formatData(dataset, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    ground_truth_labels = dataset.y
    labels = np.argmax(ground_truth_labels, axis=1)

    labels_to_be_masked = rng.choice(np.arange(len(labels)),int(len(labels)*.7),replace=False)
    masked_labels=[]
    for i in np.arange(len(labels)):
        if i in labels_to_be_masked:
            masked_labels.append(-1)
        else:
            masked_labels.append(labels[i])
    masked_labels=np.array(masked_labels)

    label_mask = masked_labels != -1

    X = dataset.x
    A = dataset.a
    G = convert_to_networkx(A)

    print("Adjacency Matrix Shape:", A.shape)
    print("Graph Nodes:", G.number_of_nodes())
    print("Graph Edges:", G.number_of_edges())

    # Convert your preprocessed data into a PyTorch Geometric Data object
    X_py = Data(
        x=torch.tensor(X, dtype=torch.float),  # Node features,
        nodes=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(np.array(A.nonzero()), dtype=torch.long),  # Edge indices
        y=torch.tensor(labels, dtype=torch.long)  # Labels
    )

    # Ensure edge_index is in the correct shape (2, num_edges)
    X_py.edge_index = X_py.edge_index.to(torch.long)

    return X_py, G, A, labels, label_mask, masked_labels, ground_truth_labels, labels_to_be_masked

def pickCoeffs(X, rng):

    coeffs = rng.random(X.shape[1])
    Wx = X @ coeffs
    coeffs /= np.std(Wx, 0)
    return coeffs

def fitIntercept(X: np.ndarray, coeffs: np.ndarray, p: float = 0.7, rng=None):

    f = lambda x: expit(X @ coeffs + x).mean().item() - p
    return root_scalar(f, method='bisect', bracket=[-50, 50]).root

def createMask(X, labels, p: float = 0.7, mech: str = 'MAR', rng=None):

    assert mech in ['MAR', 'MNAR'], "Mechanism must be one of MAR or MNAR for self-masking"

    n = labels.shape[0]

    if mech == 'MNAR':
        X = np.hstack((X, labels[:, None]))

    coeffs = pickCoeffs(X, rng)
    intercepts = fitIntercept(X, coeffs, p, rng)

    ps = expit(X @ coeffs + intercepts).squeeze()

    ber = rng.random(n)

    return np.where(ber < ps)[0]

def formatDataMissing(dataset, rate=0.7, mech='MCAR', rng=None):

    assert mech in ['MCAR', 'MAR', 'MNAR'], "Mechanism must be one of MCAR, MAR, or MNAR"
    assert rate >= 0 and rate < 1, "Rate must be positive and less than 1"

    if rng is None:
        rng = np.random.default_rng()

    ground_truth_labels = dataset.y
    labels = np.argmax(ground_truth_labels, axis=1)

    X = dataset.x
    A = dataset.a
    G = convert_to_networkx(A)

    if mech=='MCAR':
        labels_to_be_masked = rng.choice(np.arange(len(labels)),int(len(labels)*rate),replace=False)
    else:
        labels_to_be_masked = createMask(X, labels, rate, mech, rng)

    masked_labels=[]
    for i in np.arange(len(labels)):
        if i in labels_to_be_masked:
            masked_labels.append(-1)
        else:
            masked_labels.append(labels[i])
    masked_labels=np.array(masked_labels)

    label_mask = masked_labels != -1

    

    print("Adjacency Matrix Shape:", A.shape)
    print("Graph Nodes:", G.number_of_nodes())
    print("Graph Edges:", G.number_of_edges())

    # Convert your preprocessed data into a PyTorch Geometric Data object
    X_py = Data(
        x=torch.tensor(X, dtype=torch.float),  # Node features,
        nodes=torch.tensor(X, dtype=torch.float),
        edge_index=torch.tensor(np.array(A.nonzero()), dtype=torch.long),  # Edge indices
        y=torch.tensor(labels, dtype=torch.long)  # Labels
    )

    # Ensure edge_index is in the correct shape (2, num_edges)
    X_py.edge_index = X_py.edge_index.to(torch.long)

    return X_py, G, A, labels, label_mask, masked_labels, ground_truth_labels, labels_to_be_masked