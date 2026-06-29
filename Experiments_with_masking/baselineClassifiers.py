import tensorflow as tf
from spektral.layers import GCNConv, GATConv
from spektral.layers import GraphSageConv
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

def evaluate_model(true_labels, predicted_labels):
    """
    Evaluate the model's performance using accuracy, F1-score, and confusion matrix.

    Args:
        true_labels (np.array): Ground truth labels (integers).
        predicted_labels (np.array): Predicted labels (integers).

    Returns:
        dict: A dictionary containing accuracy, F1-score, and confusion matrix.
    """
    # Compute accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Compute F1-score (macro-averaged)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    #
    print(cm)
    
    # Return results as a dictionary
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }

class GCN(tf.keras.Model):
    def __init__(self, n_labels, seed=42):  # Use an explicit seed
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)  # Define initializer
        
        self.conv1 = GCNConv(16, activation='relu', kernel_initializer=initializer)
        self.conv2 = GCNConv(n_labels, activation='softmax', kernel_initializer=initializer)

    def call(self, inputs):
        x, a = inputs
        intermediate_embeddings = self.conv1([x, a])  # Store intermediate embeddings
        x = self.conv2([intermediate_embeddings, a])
        return x, intermediate_embeddings  # Return both final output and intermediate embeddings
    
class GAT(tf.keras.Model):
    def __init__(self, n_labels, num_heads=8, seed=42):
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        self.conv1 = GATConv(16, attn_heads=num_heads, concat_heads=True, activation='elu', kernel_initializer=initializer)
        self.conv2 = GATConv(n_labels, attn_heads=1, concat_heads=False, activation='softmax', kernel_initializer=initializer)

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

    def call(self, inputs):
        x, a = inputs
        intermediate_embeddings = self.conv1([x, a])  # Store intermediate embeddings
        x = self.conv2([intermediate_embeddings, a])
        return x, intermediate_embeddings  # Return both final output and intermediate embeddings
    
class MLP(tf.keras.Model):
    def __init__(self, n_labels, hidden_dim=16, depth=2, seed=42):
        super().__init__()
        assert depth > 1, "Depth must be greater than 1"
        assert n_labels > 1, "Number of labels must be greater than 1"
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        self.net = [tf.keras.layers.Dense(hidden_dim, activation='relu', kernel_initializer=initializer) for _ in range(depth)]
        if n_labels > 2:
            self.net.append(tf.keras.layers.Dense(n_labels, activation='softmax', kernel_initializer=initializer))
        else:
            self.net.append(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer))

    def call(self, inputs):
        x, a = inputs
        for layer in self.net[:-1]:
            x = layer(x)
        return self.net[-1](x), x