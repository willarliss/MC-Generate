import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array, check_X_y
from sklearn.base import BaseEstimator, ClassifierMixin

from .mc_corpus import CorpusGraph


class MarkovChainCorpus(BaseEstimator, ClassifierMixin):

    def __init__(self, ):

        self.le_ = LabelEncoder()
        self.graphs_ = {}

    def _validate_data(self, X, y=None):

        params = dict(dtype=str, ensure_2d=False)

        if y is None:
            return check_array(X, **params)

        return check_X_y(X, y, **params)

    def fit(self, X, y):

        X, y = self._validate_data(X, y)

        self.le_.fit(y)
        y = self.le_.transform(y)

        for c in self.le_.classes_:
            documents = X[y==self.le_.transform([c])]

            self.graphs_[c] = CorpusGraph()
            self.graphs_[c].add_documents(documents)

        return self

    def partial_fit(self, X, y):

        if len(self.graphs_) == 0:
            self.fit(X, y)

        else:
            X, y = self._validate_data(X, y)
            y = self.le_.transform(y)

            for c in self.le_.classes_:
                documents = X[y==self.le_.transform([c])]
                self.graphs_[c].add_documents(documents)

        return self

    def predict_proba(self, X):

        X = self._validate_data(X)

        probas = []
        for document in X:

            entropy = []
            for c in self.le_.classes_:

                entropy.append(
                    self.graphs_[c].entropy(document)
                )

            probas.append(
                [e/sum(entropy) for e in entropy]
            )

        return np.array(probas)

    def predict(self, X):

        return self.le_.inverse_transform(
            self.predict_proba(X).argmax(1)
        )
