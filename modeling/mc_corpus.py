import numpy as np
import networkx as nx
import scipy.linalg as la
import matplotlib.pyplot as plt

from .utils import iter_edges

# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name


class CorpusGraph(nx.DiGraph):

    def __init__(self, *args, delim=' ', terminal='||', **kwargs):

        super().__init__(*args, **kwargs)

        self.origins = {}

        self.delim = str(delim)
        self.terminal = str(terminal)

    def _update_edge_counts(self, v0, v1):

        if self.has_edge(v0, v1):
            self[v0][v1]['count'] += 1
        else:
            self.add_edge(v0, v1, count=1, proba=0)

    def _update_edge_probas(self, v0):

        total = self.out_degree(weight='count')[v0]

        for v1 in self[v0]:
            self[v0][v1]['proba'] = self[v0][v1]['count'] / total

    def _sample_origins(self, rng):

        nodes = np.array(list(self.origins.keys()))
        counts = np.array(list(self.origins.values()))

        return rng.choice(nodes, p=counts/counts.sum())

    def _sample_connections(self, node, rng):

        edge_data = self[node]
        edges = np.array(list(edge_data.keys()))
        probas = np.array([e['proba'] for e in edge_data.values()])

        return rng.choice(edges, p=probas)

    def _greedy_connections(self, node):

        return max(self[node].items(), key=lambda x: x[1]['count'])[0]

    def _random_walk(self, origin, seed):

        rng = np.random.default_rng(seed)

        if not origin:
            node = self._sample_origins(rng)
        else:
            node = origin

        nodes = []
        while True:
            if node == self.terminal:
                return nodes

            nodes.append(node)
            node = self._sample_connections(node, rng)

    def _greedy_walk(self, origin, seed):

        rng = np.random.default_rng(seed)

        if not origin:
            node = self._sample_origins(rng)
        else:
            node = origin

        nodes = []
        while True:
            if node == self.terminal:
                return nodes

            nodes.append(node)
            node = self._greedy_connections(node)

    def _calculate_entropy(self, path, eps=1e-9):

        entropy = []
        for v0, v1 in iter_edges(path):

            if self.has_edge(v0, v1):
                proba = self[v0][v1]['proba']
                entropy.append(np.log(proba)*proba)
            else:
                entropy.append(np.log(eps)*eps)

        return -np.sum(entropy)

    def _update_nodes(self, nodes):

        counts = self.nodes.data()

        for idx, node in enumerate(nodes):
            if idx == 0:
                try:
                    self.origins[node] += 1
                except KeyError:
                    self.origins[node] = 1

            if self.has_node(node):
                counts[node]['count'] += 1
            else:
                self.add_node(node, count=1)

    def sample(self, n, stochastic=True, start=None, seed=None):

        documents = []
        for _ in range(n):

            if stochastic:
                document = self._random_walk(start, seed)
            else:
                document = self._greedy_walk(start, seed)

            documents.append(self.delim.join(document))

        return documents

    def display(self, size=(6,6), seed=None, **kwargs):

        pos = nx.spring_layout(self, seed=seed)
        edges = {k: round(v['proba'], 2) for k, v in self.edges.items()}

        plt.figure(figsize=size)
        plt.axis('off')

        nx.draw_networkx(self, pos=pos, with_labels=True, **kwargs)
        nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edges)

        plt.show()

    def add_documents(self, documents):

        for document in documents:
            self.add_document(document)

        return self

    def add_document(self, document):

        doc_split = (document+self.delim+self.terminal).split(self.delim)

        self._update_nodes(doc_split)

        for v0, v1 in iter_edges(doc_split):
            self._update_edge_counts(v0, v1)
            self._update_edge_probas(v0)

        return self

    def entropy(self, document):

        doc_split = (document+self.delim+self.terminal).split(self.delim)

        return self._calculate_entropy(doc_split)

    def transition_matrix(self, nodes=None):

        if nodes is None:
            nodes = list(self.nodes)

        return np.asarray(
            nx.adjacency_matrix(self.subgraph(nodes), weight='proba').todense()
        )

    def perron_vector(self, nodes=None):

        P = self.transition_matrix(nodes)

        eigen = la.eig(P, left=True, right=False)

        return eigen[1][:, eigen[0].argmax()]

    def laplacian_matrix(self, nodes=None):

        P = self.transition_matrix(nodes)
        Phi = np.diag(self.perron_vector())

        Phi_a = la.fractional_matrix_power(Phi, 1/2)
        Phi_b = la.fractional_matrix_power(Phi, -1/2)
        P_ = P.conj().T

        return (Phi_a.dot(P).dot(Phi_b) + Phi_b.dot(P_).dot(Phi_a)).real / 2

    def prune(self, cutoff=1, inplace=False):

        raise NotImplementedError