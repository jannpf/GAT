import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, DeepGraphInfomax


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.6):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels,
            8,
            heads=8,
            dropout=self.dropout)
        self.conv2 = GATConv(
            8 * 8,
            out_channels,
            heads=1,
            concat=False,
            dropout=self.dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Readout function (simple global mean of node embeddings)


def summary(x, *args, **kwargs):
    return torch.sigmoid(x.mean(dim=0))

# Define the corruption function (node feature shuffling)


def corruption(x, edge_index, edge_attr=None):
    return x[torch.randperm(x.size(0))], edge_index, edge_attr

# Create the DGI model


class DGIModel(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.6):
        super(DGIModel, self).__init__()
        self.encoder = GATEncoder(in_channels, out_channels, dropout=dropout)
        self.dgi = DeepGraphInfomax(
            hidden_channels=out_channels,
            encoder=self.encoder,
            summary=summary,
            corruption=corruption
        )

    def forward(self, data):
        return self.dgi(data.x, data.edge_index, data.edge_attr)
