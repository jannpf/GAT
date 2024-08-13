from collections import defaultdict, Counter

import igraph as ig
import leidenalg as la
import networkx as nx
from sklearn.cluster import KMeans, OPTICS
# from sklearn.metrics import silhouette_score

from . import community_results


def _combine_labels(G, labels1, labels2):
    combined_labels = {}
    for node in G.nodes():
        combined_labels[node] = (labels1[node], labels2[node])
    return combined_labels


def _propagate_labels(G, labels):
    new_labels = labels.copy()
    while True:
        unchanged = True
        for node in G.nodes():
            label_count = Counter()
            for neighbor in G.neighbors(node):
                label_count[new_labels[neighbor]] += 1
            # Get the label with the maximum count
            most_common_label = max(label_count, key=label_count.get)
            if new_labels[node] != most_common_label:
                new_labels[node] = most_common_label
                unchanged = False
        if unchanged:
            break
    return new_labels


def _aggregate_solutions(G, num_runs):
    assert num_runs > 0, ValueError('num_runs set to zero')
    
    for i in range(num_runs):
        # Run label propagation to get a new solution
        solution = list(nx.algorithms.community.label_propagation_communities(G))
        labels = {node: label for label, community in enumerate(solution) for node in community}
        # Combine the current aggregate labels with the new solution labels
        if i == 0:
            combined_labels = labels
        else:
            combined_labels = _combine_labels(G, combined_labels, labels)
            combined_labels = _propagate_labels(G, combined_labels)
    
    # Convert the combined labels to final communities
    unique_labels = {label: idx for idx, label in enumerate(set(combined_labels.values()))}
    final_partition = {node: unique_labels[label] for node, label in combined_labels.items()}
    
    return final_partition


def label_propagation(g, aggr_runs=None):
    if aggr_runs is None:
        communities = nx.algorithms.community.label_propagation_communities(g)
        labels = {n: i for i, c in enumerate(communities) for n in c}
    else:
        labels = _aggregate_solutions(g, aggr_runs)
    return labels


def louvain(g, resolution=1):
    communities = nx.algorithms.community.louvain_communities(g, resolution=resolution)
    labels = {n: i for i, c in enumerate(communities) for n in c}

    return labels


def leiden(g: nx.Graph, n_iterations: int = -1, seed=None, resolution=None):
    ig_graph = ig.Graph.from_networkx(g)
    if resolution is not None:
        leiden_partition = la.find_partition(
            ig_graph,
            partition_type=la.CPMVertexPartition,
            resolution_parameter=resolution,
            n_iterations=n_iterations,
            seed=seed,
        )
    else:
        leiden_partition = la.find_partition(
            ig_graph,
            partition_type=la.ModularityVertexPartition,
            n_iterations=n_iterations,
            seed=seed,
        )

    return {node: part for node, part in zip(g.nodes(), leiden_partition.membership)}


def kmeans(g, node_embeddings, max_num_clusters=14):
    """
    Use k-means clustering on node embeddings
    for range [2, max_num_clusters].
    Best clustering is selected based on max modularity.
    """
    best_score = -1
    best_k = 2
    for k in range(2, max_num_clusters):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(node_embeddings)
        labels = list(kmeans.labels_)
        communities = {node: label for node, label in enumerate(labels)}
        score = community_results.community_metrics(g, communities)["Modularity"]
        # alternatively
        # score = silhouette_score(node_embeddings, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k

    kmeans = KMeans(n_clusters=best_k, random_state=0).fit(node_embeddings)
    labels = list(kmeans.labels_)
    communities = {node: label for node, label in enumerate(labels)}
    scores = community_results.community_metrics(g, communities)
    return scores, best_k, communities


def optics(g, node_embeddings):
    """
    Use optics clustering on node embeddings.
    Best clustering is selected based on max modularity.
    """
    optics = OPTICS(min_samples=5)
    clusters = optics.fit_predict(node_embeddings)
    clusters = clusters + 1  # default starts with -1
    communities = dict(zip(range(g.number_of_nodes()), clusters))
    best_k = len(set(clusters))
    scores = community_results.community_metrics(g, communities)
    return scores, best_k, communities
