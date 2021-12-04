import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name


def iter_edges(nodes):
    for idx in range(len(nodes)-1):
        yield nodes[idx], nodes[idx+1]


def membership(graphs, document):

    entropies = []

    for graph in graphs:
        entropies.append(
            graph.entropy(document)
        )

    return [e/sum(entropies) for e in entropies]


def perron_vector(G, nodes=None, random_state=None):

    P = G.transition_matrix(nodes, sparse=True)
    shape = P.shape[0]

    eigen = sp.linalg.eigs(
        A=P.T,
        tol=0.,
        which='LM',
        k=np.floor(np.log(shape)).astype(int),
        v0=np.random.default_rng(random_state).normal(size=shape),
    )

    return eigen[1][:, eigen[0].argmax()]


def laplacian_matrix(G, nodes=None, random_state=None, sparse=False):

    P = G.transition_matrix(nodes, sparse=True)
    Phi = perron_vector(G, nodes=nodes, random_state=random_state)

    Phi_a = sp.csr_matrix(la.fractional_matrix_power(np.diag(Phi), 1/2))
    Phi_b = sp.csr_matrix(la.fractional_matrix_power(np.diag(Phi), -1/2))

    P_ = P.conj().T

    L = (Phi_a.dot(P).dot(Phi_b) + Phi_b.dot(P_).dot(Phi_a)).real / 2

    if sparse:
        return L

    return np.asarray(L.todense())


def truncated_svd(X, trunc=None):

    X = sp.csc_matrix(X, dtype=float)
    U, s, Vt = sp.linalg.svds(X)

    if trunc is None:
        trunc = sum(s>s.mean())
    elif isinstance(trunc, float):
        trunc = int(s.shape[0]*trunc)
    else:
        trunc = int(trunc)

    return (
        U[:, :trunc],
        np.diag(s[:trunc]),
        Vt[:trunc, :],
    )


def cluster_svd_sign(X, max_clusters=8):

    trunc = np.floor(np.log2(max_clusters)).astype(int)

    singular_vecs = truncated_svd(X, trunc)[0]

    unique = np.unique(singular_vecs>0, axis=0)

    labels = []
    for vec in singular_vecs>0:

        mask = (unique==vec).all(axis=1)
        lab = np.where(mask)[0]
        labels.append(lab)

    return np.squeeze(labels)


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
