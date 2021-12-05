"""Probabilistic directed graph for text corpu"""

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix

from .utils import iter_edges


class CorpusGraph(nx.DiGraph):

    def __init__(self, *args, delim=' ', terminal='||', **kwargs):

        super().__init__(*args, **kwargs)

        self.origins = {}

        self.delim = str(delim)
        self.terminal = str(terminal)

    def _update_edge_probas(self):

        nodes = list(self.nodes)

        A = nx.adjacency_matrix(self.subgraph(nodes), weight='count')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            A = A / A.sum(1)

        update_dict = {}
        for key, value in dok_matrix(A).items():

            edge = (nodes[key[0]], nodes[key[1]])
            if self.has_edge(*edge):
                update_dict[edge] = {'proba': value}

        nx.set_edge_attributes(self, update_dict)

    def _update_edges(self, v0, v1):

        if self.has_edge(v0, v1):
            self[v0][v1]['count'] += 1
        else:
            self.add_edge(v0, v1, count=1, proba=0)

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

        all_nodes = []

        for document in documents:
            doc_split = (document+self.delim+self.terminal).split(self.delim)

            all_nodes.extend(doc_split)
            self._update_nodes(doc_split)

            for v0, v1 in iter_edges(doc_split):
                self._update_edges(v0, v1)

        self._update_edge_probas()

        return self

    def add_document(self, document):

        doc_split = (document+self.delim+self.terminal).split(self.delim)

        self._update_nodes(doc_split)

        for v0, v1 in iter_edges(doc_split):
            self._update_edges(v0, v1)

        self._update_edge_probas()

        return self

    def entropy(self, document):

        doc_split = (document+self.delim+self.terminal).split(self.delim)

        return self._calculate_entropy(doc_split)

    def transition_matrix(self, nodes=None, sparse=False):

        if nodes is None:
            nodes = list(self.nodes)

        adj = nx.adjacency_matrix(self.subgraph(nodes), weight='proba')

        if sparse:
            return adj

        return np.asarray(adj.todense())
