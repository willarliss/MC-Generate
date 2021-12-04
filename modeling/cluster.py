from sklearn.utils import check_array
from sklearn.base import BaseEstimator, ClassifierMixin

from .mc_corpus import CorpusGraph
from .utils import laplacian_matrix, cluster_svd_sign, prune_graph

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name


class MarkovSVDClustering(BaseEstimator, ClassifierMixin):

    def __init__(self, max_clusters=8, prune=0):

        self.max_clusters = max_clusters
        self.prune = prune
        self.G = CorpusGraph()

    def _validate_data(self, X):

        return check_array(X, dtype=str, ensure_2d=False)

    def fit(self, documents, y=None):

        documents = self._validate_data(documents)

        self.G.add_documents(documents)

        if self.prune > 0:
            self.G = prune_graph(self.G, cutoff=self.prune)

        return self

    def predict(self, nodes=None):

        if nodes is None:
            nodes = list(self.G.nodes)

        L = laplacian_matrix(self.G, nodes=nodes)

        labels = cluster_svd_sign(L, max_clusters=self.max_clusters)

        return dict(zip(nodes, labels))
