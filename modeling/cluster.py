import numpy as np
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, ClassifierMixin

from .mc_corpus import CorpusGraph
from .utils import laplacian_matrix, cluster_svd_sign


class MarkovSVDClustering(BaseEstimator, ClassifierMixin):

    def __init__(self, max_clusters=8):

        self.max_clusters = max_clusters
        self.G = CorpusGraph()

    def _validate_data(self, X):

        return check_array(X, dtype=str, ensure_2d=False)

    def fit(self, documents, y=None):

        documents = self._validate_data(documents)

        self.G.add_documents(documents)

        return self

    def predict(self, nodes=None):

        if nodes is None:
            nodes = list(self.G.nodes)

        L = laplacian_matrix(self.G, nodes=nodes)

        labels = cluster_svd_sign(L, max_clusters=self.max_clusters)

        return dict(zip(nodes, labels))
