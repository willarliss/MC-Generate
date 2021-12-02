import numpy as np
import networkx as nx
import scipy.linalg as la


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


def perron_vector(G, nodes=None):

    P = G.transition_matrix(nodes)

    eigen = la.eig(P, left=True, right=False)

    return eigen[1][:, eigen[0].argmax()]


def laplacian_matrix(G, nodes=None):

    P = G.transition_matrix(nodes)
    Phi = np.diag(perron_vector(G, nodes))

    Phi_a = la.fractional_matrix_power(Phi, 1/2)
    Phi_b = la.fractional_matrix_power(Phi, -1/2)
    P_ = P.conj().T

    return (Phi_a.dot(P).dot(Phi_b) + Phi_b.dot(P_).dot(Phi_a)).real / 2


def truncated_svd(X, trunc=None):

    U, s, Vt = la.svd(X)

    if trunc is None:
        trunc = sum(s>s.mean())
    if isinstance(trunc, float):
        trunc = int(s.shape[0]*trunc)

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
