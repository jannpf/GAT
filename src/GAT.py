import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    """
    A higher level implementation of the whole GAT model.
    """

    def __init__(self, num_features, hidden_channels, out_channels, num_heads, dropout):
        super(GAT, self).__init__()
        self.dropout = dropout
        print(dropout)
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def contrastive_loss(output, G, margin=1.0):
    """
    A loss function to use with GAT.
    Idea: make embeddings of connected nodes as similar as possible,
    make embeddings of unconnected nodes as different as possible.
    On each iteration, sample max 10000 pairs of unconnected nodes,
    update embeddings based on this info.
    """

    # positive == connected nodes
    positive_pairs = np.array(list(G.edges()))
    num_nodes = G.number_of_nodes()

    # negative = unconnected nodes
    # Random sampling of negative pairs
    negative_pairs = []
    nodes = list(G.nodes())
    max_num_pairs = num_nodes ** 2
    num_negative_samples = min(1e4, max_num_pairs)

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