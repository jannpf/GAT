import argparse

import networkx as nx
import pandas as pd
import torch
import torch.optim as optim
from scipy.io import mmread
from torch_geometric.data import Data

from .community_results import community_metrics, plot_communities
from .clustering import kmeans, optics
from . import GAT


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
    model = GAT.GAT(num_features, HIDDEN_CHANNELS, OUT_CHANNELS, NUM_HEADS, P_DROPOUT).to(DEVICE)

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
            loss = GAT.contrastive_loss(out, G)
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
    _, best_k, labels = optics(G, node_embeddings)

    # Calculations and visualizations
    metrics = community_metrics(G, labels)
    print(metrics)
    plot_communities(G, labels)
