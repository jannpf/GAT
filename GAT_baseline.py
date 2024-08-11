import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import networkx.algorithms.community as nx_comm
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import mmread
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix


HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32  # Size of the embedding
NUM_HEADS = 8
LR = 0.01
P_DROPOUT = 0.6
NUM_EPOCHS = 100
MAX_NUM_CLUSTERS = 12
DATA_PATH = "./data/"


def load_data(dataset):
    # Load the dataset
    if dataset == "karate":
        G = nx.karate_club_graph()
    elif dataset == "enron":
        # Load the Matrix Market file
        matrix = mmread(DATA_PATH + "email_enron_only.mtx")
        G = nx.from_scipy_sparse_array(matrix)
    elif dataset == "univ":
        # Convert the sparse matrix to a NetworkX graph
        G = nx.read_edgelist(DATA_PATH + "email_univ.edges")
        G = nx.from_numpy_array(nx.adjacency_matrix(G).todense())
    elif dataset == "deezer":
        G = pd.read_csv(DATA_PATH + "deezer_clean_data/" + "HR_edges.csv")
        G = nx.from_pandas_edgelist(G, source="node_1", target="node_2")
    else:
        raise Exception("No valid dataset specified")
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    adj_matrix = nx.adjacency_matrix(G).todense()
    return G, edge_index, adj_matrix


class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=P_DROPOUT, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=P_DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def contrastive_loss(output, G, margin=1.0):
    positive_pairs = []
    negative_pairs = []

    # Create positive pairs (connected nodes)
    for edge in G.edges():
        u, v = edge
        positive_pairs.append((u, v))

    # Create negative pairs (unconnected nodes)
    nodes = list(G.nodes())
    for u in nodes:
        for v in nodes:
            if u != v and not G.has_edge(u, v):
                negative_pairs.append((u, v))

    # Compute the loss
    loss = 0.0
    for u, v in positive_pairs:
        dist = torch.norm(output[u] - output[v])
        loss += dist**2

    for u, v in negative_pairs:
        dist = torch.norm(output[u] - output[v])
        loss += torch.clamp(margin - dist, min=0.0)**2

    return loss / (len(positive_pairs) + len(negative_pairs))


def get_kmeans_pred(node_embeddings):
    # Use k-means clustering
    best_score = -1
    best_k = 2
    for k in range(2, MAX_NUM_CLUSTERS):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(node_embeddings)
        score = silhouette_score(node_embeddings, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(node_embeddings)
    labels = kmeans.labels_
    return best_k, labels


def plot_communities(G, node_community_labels, title="Communities"):
    plt.figure(figsize=(15, 7))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G, pos, node_color=node_community_labels, with_labels=True, cmap=plt.cm.Set3
    )
    plt.title(title)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a dataset")
    parser.add_argument(
        "-d",
        type=str,
        choices=["karate", "univ", "enron", "deezer"],
        required=True,
        help="Specify the dataset: karate, univ, enron, or deezer",
    )
    parser.add_argument(
        "-l",
        type=str,
        choices=["variance", "contrastive"],
        required=False,
        default="variance",
        help="Specify the loss func: variance or contrastive",
    )
    args = parser.parse_args()
    DATASET = args.d
    LOSS = args.l

    # load data
    G, edge_index, adj_matrix = load_data(DATASET)

    # Create a PyTorch Geometric data object
    data = Data(edge_index=edge_index)
    data.num_nodes = G.number_of_nodes()

    # Print some basic info
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")

    # Initialize the GAT model
    num_features = data.num_nodes  # We'll use one-hot encodings of nodes as features
    model = GAT(num_features, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_HEADS)

    # Initialize features as identity matrix (one-hot encoding of nodes)
    data.x = torch.eye(data.num_nodes)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    # Define the training loop
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        if LOSS == "variance":
            # Since the task is unsupervised, we'll minimize the variance of node embeddings
            loss = -torch.var(out)
        else:
            loss = contrastive_loss(out, G)
        loss.backward()
        optimizer.step()
        return loss.item()

    # Training process
    for epoch in range(NUM_EPOCHS):
        loss = train()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Get the node embeddings
    model.eval()
    node_embeddings = model(data).detach().numpy()

    # get kmeans predictions
    best_k, labels = get_kmeans_pred(node_embeddings)

    # Convert labels to communities
    communities = [[] for _ in range(best_k)]
    for node, label in enumerate(labels):
        communities[label].append(node)

    modularity = nx_comm.modularity(G, communities)
    print(f"Best number of communities: {best_k}, Modularity: {modularity}")

    plot_communities(G, labels)
