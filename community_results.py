import numpy as np
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

    return {
        "Number of Communities": num_communities,
        "Average Community Size": avg_community_size,
        "Modularity": mod,
        "Coverage": coverage,
        "Performance": performance,
    }


def plot_communities(G, node_community_labels, title="Communities"):
    plt.figure(figsize=(15, 7))
    pos = nx.spring_layout(G, seed=42)

    if not isinstance(node_community_labels, np.ndarray):
        node_community_labels = list(node_community_labels.values())
    nx.draw(G, pos, node_color=node_community_labels,
            with_labels=True, cmap=plt.cm.Set3)
    plt.title(title)

    plt.show()
