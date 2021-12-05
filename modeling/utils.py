"""Utility functions"""

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


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


def eigen_solver(A, k=None, random_state=None):

    shape = A.shape[0]

    return sp.linalg.eigs(
        A=A,
        tol=0.,
        which='LM',
        maxiter=shape*10,
        k=(int(np.log(shape)) if k is None else int(k)),
        v0=np.random.default_rng(random_state).normal(size=shape),
    )


def singular_value_solver(A, k=None, random_state=None):

    shape = A.shape[0]

    return sp.linalg.svds(
        A=A,
        tol=0.,
        which='LM',
        solver='arpack',
        maxiter=shape*10,
        k=(int(np.log(shape)) if k is None else int(k)),
        v0=np.random.default_rng(random_state).normal(size=shape),
    )


def perron_vector(G, nodes=None, random_state=None):

    P = G.transition_matrix(nodes, sparse=True)

    eig = eigen_solver(P.T, random_state=random_state)

    return eig[1][:, eig[0].argmax()]


def laplacian_matrix(G, nodes=None, random_state=None, sparse=False):
    """http://www.math.ucsd.edu/~fan/wp/dichee.pdf"""

    P = G.transition_matrix(nodes, sparse=True)
    Phi = perron_vector(G, nodes=nodes, random_state=random_state)

    Phi_a = sp.csr_matrix(la.fractional_matrix_power(np.diag(Phi), 1/2))
    Phi_b = sp.csr_matrix(la.fractional_matrix_power(np.diag(Phi), -1/2))

    P_ = P.conj().T

    L = Phi_a.dot(P).dot(Phi_b) + Phi_b.dot(P_).dot(Phi_a)
    L = sp.eye(L.shape[0]) - (L.real/2)

    if sparse:
        return L

    return np.asarray(L.todense())


def truncated_ed(A, trunc=None, random_state=None):

    A = sp.csc_matrix(A, dtype=float)
    shape = A.shape[0]

    if trunc is None:
        trunc_k = shape-2
    elif isinstance(trunc, float):
        trunc_k = int(shape*trunc)
    else:
        trunc_k = int(trunc)

    e, Vk = eigen_solver(A, k=trunc_k, random_state=random_state)

    sort = np.argsort(e)[::-1]
    start = sum(e.real==0)

    if trunc is None:
        trunc = sum(e>e.mean()) + start
    else:
        trunc = trunc_k + start

    return (
        Vk[:, sort][:, start:trunc],
        np.diag(e[sort][start:trunc])
    )


def truncated_svd(A, trunc=None, random_state=None):

    A = sp.csc_matrix(A, dtype=float)

    if trunc is None:
        trunc_k = min(A.shape)-1
    elif isinstance(trunc, float):
        trunc_k = int(A.shape[0]*trunc)
    else:
        trunc_k = int(trunc)

    U, s, Vt = singular_value_solver(A, k=trunc_k, random_state=random_state)

    if trunc is None:
        trunc = sum(s>s.mean())
    else:
        trunc = trunc_k

    return (
        U[:, :trunc],
        np.diag(s[:trunc]),
        Vt[:trunc, :],
    )


def extended_fiedler_method(L, max_clusters=8, decomposition='svd', random_state=None):
    """http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.628.7748&rep=rep1&type=pdf"""

    trunc = np.floor(np.log2(max_clusters)).astype(int)

    if decomposition == 'svd':
        singular_vecs = truncated_svd(L, trunc=trunc, random_state=random_state)[0]
    elif decomposition == 'ed':
        singular_vecs = truncated_ed(L, trunc=trunc, random_state=random_state)[0]
    else:
        raise ValueError(f'Unknown decomposition method: {decomposition}')

    #assert singular_vecs.shape[0] == L.shape[0]
    #assert 2**singular_vecs.shape[1] <= max_clusters

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
