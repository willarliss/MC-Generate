"""Graph clustering with extended Fiedler method"""

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

import warnings

from sklearn.utils import check_array
from sklearn.base import BaseEstimator, ClusterMixin

from .graph import CorpusGraph
from .utils.linalg import laplacian_matrix, extended_fiedler_method
from .utils.gutils import prune_graph


class SpectralNodeClustering(BaseEstimator, ClusterMixin):

    def __init__(self, max_clusters=8, prune=0, method='svd', random_state=None):

        self.max_clusters = max_clusters
        self.prune = prune
        self.method = method
        self.random_state = random_state
        self.graph_ = CorpusGraph()

    def _validate_data(self, X):

        return check_array(X, dtype=str, ensure_2d=False)

    def fit(self, documents):

        documents = self._validate_data(documents)

        self.graph_.add_documents(documents)

        if self.prune > 0:
            warnings.warn('modeling.utils.gutils.prune_graph does not work as it should. Returns disconnected graph.')
            self.graph_ = prune_graph(self.graph_, cutoff=self.prune)

        return self

    def fit_predict(self, documents, nodes=None):

        self.fit(documents)

        if nodes is None:
            nodes = list(self.graph_.nodes)

        L = laplacian_matrix(self.graph_, nodes=nodes)

        labels = extended_fiedler_method(
            L=L,
            max_clusters=self.max_clusters,
            decomposition=self.method,
            random_state=self.random_state,
        )

        return dict(zip(nodes, labels))
