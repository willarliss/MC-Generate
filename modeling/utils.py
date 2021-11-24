def iter_edges(nodes):
    for idx in range(len(nodes)-1):
        yield nodes[idx], nodes[idx+1]

def membership(graphs, document):

    entropies = []

    for graph in graphs:
        entropies.append(
            graph.entropy(document)
        )

    #entropies = np.exp(entropies)
    #return entropies/entropies.sum()
    return [e/sum(entropies) for e in entropies]
