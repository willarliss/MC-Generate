"""Probabilistic directed graph for text corpus"""

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix

from .utils.gutils import iter_edges


class CorpusGraph(nx.DiGraph):

    max_walk = 100
    terminal = '||'

    def __init__(self, incoming_graph_data=None, delim=' ', **attr):

        super().__init__(incoming_graph_data, **attr)

        self.delim = str(delim)

    def _update_edge_probas(self):

        nodes = list(self.nodes)

        A = nx.adjacency_matrix(self.subgraph(nodes), weight='count')

        with warnings.catch_warnings():
            # Catch divide by zero warning associated with self.terminal node
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
            if self.has_node(node):
                counts[node]['count'] += 1
                counts[node]['origin'] += int(idx==0)
            else:
                self.add_node(node, count=1, origin=int(idx==0))

    def _sample_origins(self, rng):

        nodes = nx.get_node_attributes(self, 'origin')
        nodes, probas = np.array(list(nodes.keys())), np.array(list(nodes.values()))
        probas = probas / probas.sum()

        return rng.choice(nodes, p=probas)

    def _sample_connections(self, node, rng):

        edge_data = self[node]
        edges = np.array(list(edge_data.keys()))
        probas = np.array([e['proba'] for e in edge_data.values()])

        return rng.choice(edges, p=probas)

    def _greedy_connections(self, node):

        edge_data = self[node]

        try:
            max(edge_data.items(), key=lambda x: x[1]['count'])[0]
        except ValueError:
            print(edge_data, node)

        return max(edge_data.items(), key=lambda x: x[1]['count'])[0]

    def _random_walk(self, origin, stochastic, rng):

        if origin is None:
            node = self._sample_origins(rng)
        else:
            node = origin

        nodes = []
        while True:

            if (node == self.terminal) or (len(nodes) > self.max_walk):
                return nodes

            nodes.append(node)

            if stochastic:
                node = self._sample_connections(node, rng)
            else:
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

    def add_documents(self, documents):

        for document in documents:
            doc_split = (document+self.delim+self.terminal).split(self.delim)

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

    def sample(self, stochastic=True, start=None, seed=None):

        rng = np.random.default_rng(seed)

        document = self._random_walk(start, stochastic, rng)

        return self.delim.join(document)

    def display(self, size=(6,6), labels=True, seed=None, **kwargs):

        pos = nx.spring_layout(self, seed=seed)

        plt.figure(figsize=size)
        plt.axis('off')

        if labels:
            edges = {k: round(v['proba'], 2) for k, v in self.edges.items()}
            nx.draw_networkx(self, pos=pos, with_labels=True, **kwargs)
            nx.draw_networkx_edge_labels(self, pos=pos, edge_labels=edges)
        else:
            nx.draw_networkx(self, pos=pos, with_labels=False, **kwargs)

        plt.show()

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


class CorpusGraph1(CorpusGraph):
    # Use origins dictionary instead of origins node attribute

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.origins = {}

    def _sample_origins(self, rng):

        nodes = np.array(list(self.origins.keys()))
        counts = np.array(list(self.origins.values()))

        return rng.choice(nodes, p=counts/counts.sum())

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
