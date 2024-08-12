import argparse

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch_geometric.data import Data

from .community_results import community_metrics, plot_communities
from .GAT import GAT


HIDDEN_CHANNELS = 64
OUT_CHANNELS = 32  # Size of the embedding
NUM_HEADS = 8
LR = 0.001
P_DROPOUT = 0.6
NUM_EPOCHS = 300
MAX_NUM_CLUSTERS = 12
DATA_PATH = "./data/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def contrastive_loss(output, G, margin=1.0):
    positive_pairs = np.array(list(G.edges()))
    num_nodes = G.number_of_nodes()

    # Random sampling of negative pairs
    negative_pairs = []
    nodes = list(G.nodes())
    num_iterations = num_nodes ** 2
    negative_sampling_rate = 1e4 / num_iterations  # num_neg_samples should be <= 1e4
    num_negative_samples = min(int(negative_sampling_rate * num_iterations), num_iterations)

    while len(negative_pairs) < num_negative_samples:
        u = np.random.choice(nodes)
        v = np.random.choice(nodes)
        if u != v and not G.has_edge(u, v):
            negative_pairs.append((u, v))

    negative_pairs = np.array(negative_pairs)

    # Compute positive loss
    positive_u = output[positive_pairs[:, 0]]
    positive_v = output[positive_pairs[:, 1]]
    positive_distances = torch.norm(positive_u - positive_v, dim=1)
    positive_loss = torch.sum(positive_distances**2)

    # Compute negative loss
    negative_u = output[negative_pairs[:, 0]]
    negative_v = output[negative_pairs[:, 1]]
    negative_distances = torch.norm(negative_u - negative_v, dim=1)
    negative_loss = torch.sum(torch.clamp(margin - negative_distances, min=0.0)**2)

    # Combine losses
    total_loss = positive_loss + negative_loss
    total_pairs = len(positive_pairs) + len(negative_pairs)

    return total_loss / total_pairs


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
    data = Data(edge_index=edge_index).to(DEVICE)
    data.num_nodes = G.number_of_nodes()
    # Initialize features as identity matrix (one-hot encoding of nodes)
    data.x = torch.eye(data.num_nodes).to(DEVICE)
    num_features = data.num_nodes  # We'll use one-hot encodings of nodes as features

    # Print some basic info
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Training model on {DEVICE}")

    # Initialize the GAT model
    model = GAT(num_features, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_HEADS, P_DROPOUT).to(DEVICE)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

    # Define the training loop
    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        if LOSS == "variance":
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

    # Calculations and visualizations
    metrics = community_metrics(G, list(labels))
    print(metrics)
    plot_communities(G, labels)
