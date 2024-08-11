import networkx as nx
import numpy as np

import leidenalg as la
import igraph as ig


def label_propagation(g):
    communities = nx.algorithms.community.label_propagation_communities(g)
    labels = {n: i for i, c in enumerate(communities) for n in c}

    return labels


def louvain(g):
    communities = nx.algorithms.community.louvain_communities(g)
    labels = {n: i for i, c in enumerate(communities) for n in c}

    return labels


def leiden(g: nx.Graph, n_iterations: int = 2, seed=None):
    ig_graph = ig.Graph.from_networkx(g)

    leiden_partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        n_iterations=n_iterations,
        seed=seed,
    )

    return {node: part for node, part in zip(g.nodes(), leiden_partition.membership)}
