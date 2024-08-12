import igraph as ig
import leidenalg as la
import networkx as nx
from sklearn.cluster import KMeans, OPTICS
# from sklearn.metrics import silhouette_score

from . import community_results


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


def kmeans(g, node_embeddings, max_num_clusters=12):
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
    return best_score, best_k, communities


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
    modularity = community_results.community_metrics(g, communities)['Modularity']
    return modularity, best_k, communities
