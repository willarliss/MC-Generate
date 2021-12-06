"""Graph utility functions"""

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def iter_edges(nodes):

    for idx in range(len(nodes)-1):
        yield nodes[idx], nodes[idx+1]


def membership(graphs, document):
    """https://math.nd.edu/assets/275279/leblanc_thesis.pdf"""

    entropies = []

    for graph in graphs:
        entropies.append(
            graph.entropy(document)
        )

    return [e/sum(entropies) for e in entropies]


def prune_graph(G, cutoff=1):

    #in_degree = dict(self.in_degree)
    out_degree =  dict(G.out_degree)

    keep = []
    for node in G.nodes:

        #if in_degree[node] > cutoff and out_degree[node] > cutoff:
        if out_degree[node] > cutoff:
            keep.append(node)

    G = G.subgraph(keep).copy()
    G._update_edge_probas()

    return G
