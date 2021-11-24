import networkx as nx
import numpy as np
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

    def prune(self, cutoff):

        raise NotImplementedError


class CorpusGraph2(CorpusGraph):
    """Using origin attribute on nodes instead origins dict"""

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        delattr(self, 'origins')

    def _sample_origins(self, rng):

        origins = nx.get_node_attributes(self, 'origin')
        nodes = np.array(list(origins.keys()))
        probas = np.array(list(origins.values())) / sum(origins.values())

        return rng.choice(nodes, p=probas)

    def _update_nodes(self, nodes):

        counts = self.nodes.data()

        for idx, node in enumerate(nodes):
            if self.has_node(node):
                counts[node]['count'] += 1
                counts[node]['origin'] += int(idx==0)
            else:
                self.add_node(node, count=1, origin=int(idx==0))

    def prune(self, cutoff):

        raise NotImplementedError
