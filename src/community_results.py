import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def community_metrics(graph, labels):
    """
    Calculate community metrics for a given graph and set of labels.

    Parameters:
    - graph: The input graph (NetworkX graph object).
    - labels: A dictionary mapping node IDs to community labels.

    Returns:
    - A dictionary containing the calculated metrics:
        - "Modularity": The modularity score of the community partition.
        - "Number of Communities": The number of detected communities.
        - "Average Community Size": The average size of the communities.
    """

    # Group nodes by community
    communities = {}
    for node, label in labels.items():
        if label not in communities:
            communities[label] = []
        communities[label].append(node)
    community_list = list(communities.values())

    # Calculate modularity
    mod = nx.algorithms.community.modularity(graph, community_list)

    # Coverage and performance
    coverage, performance = nx.algorithms.community.partition_quality(
        graph, community_list)

    # Number of communities
    num_communities = len(community_list)

    # Average community size
    avg_community_size = sum(len(community)
                             for community in community_list) / num_communities

    # Average cluster coefficients
    avg_cluster_coeff = nx.average_clustering(graph)

    # Number of connected components
    conn_comp = len(list(nx.connected_components(graph)))

    return {
        "Number of Communities": num_communities,
        "Average Community Size": avg_community_size,
        "Modularity": mod,
        "Average Cluster Coefficients": avg_cluster_coeff,
        "Number of connected components": conn_comp,
        "Coverage": coverage,
        "Performance": performance,
    }


def plot_communities(G, node_community_labels, title="Communities", plot_labels=True):
    plt.figure(figsize=(15, 7))
    pos = nx.spring_layout(G, seed=42)

    if not isinstance(node_community_labels, np.ndarray):
        node_community_labels = list(node_community_labels.values())
    if not plot_labels:
        nx.draw(G, pos, node_color=node_community_labels,
                with_labels=plot_labels, cmap=plt.cm.Set3, node_size=70)
    else:
        nx.draw(G, pos, node_color=node_community_labels,
                with_labels=True, cmap=plt.cm.Set3)
    plt.title(title)

    plt.show()


def community_size_hist(communities, title="Community size distribution"):
    communities = pd.DataFrame.from_dict(communities, orient="index", columns=["community"])
    communities.value_counts("community").sort_index().plot(kind="bar", figsize=(6, 3), title=title, rot=0)
    plt.show()
