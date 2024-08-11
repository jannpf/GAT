import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv, DeepGraphInfomax


class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=8, dropout=0.6):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads = heads
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=self.dropout))

        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout))
        
        self.layers.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=self.dropout))

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        for layer in self.layers[:-1]:
            x = F.elu(layer(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
        x = self.layers[-1](x, edge_index)
        return x


def summary(x, *args, **kwargs):
    return torch.sigmoid(x.mean(dim=0))


def corruption(x, edge_index, edge_attr=None):
    return x[torch.randperm(x.size(0))], edge_index, edge_attr


class DGIModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=8, dropout=0.6):
        super(DGIModel, self).__init__()
        
        self.encoder = GATEncoder(in_channels, hidden_channels, out_channels, num_layers, heads, dropout)
        self.dgi = DeepGraphInfomax(
            hidden_channels=out_channels,
            encoder=self.encoder,
            summary=summary,
            corruption=corruption
        )

    def forward(self, data):
        return self.dgi(data.x, data.edge_index, data.edge_attr)
