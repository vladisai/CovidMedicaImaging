import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

import data


class Model:
    N_CLASSES = 14

    def fit(self, data):
        pass

    def predict(self, data):
        pass


class Baseline(Model):
    def predict(self, x):
        return np.zeros_like(x['lab'])


class SafeOneClassMixin:
    """A hack mixin required to make
    the sklearn classifirs work with training data where for some
    classes we get all training samples with label 0.
    Maybe remove those classes?
    """
    def fit(self, X, y, **kw):
        self._single_class = False
        if len(np.unique(y)) == 1:
            self._single_class = True
            self.classes_ = np.unique(y)
            return self
        return super().fit(X, y, **kw)

    def predict(self, X):
        if self._single_class:
            return np.full(len(X), self.classes_[0])
        return super().predict(X)

    def predict_proba(self, X):
        if self._single_class:
            result = np.zeros((len(X), 2))
            result[:, self.classes_.astype(np.int)[0]] = 1
            return result
        return super().predict_proba(X)


class LinearRegression(Model):
    class SafeOneClassLogisticRegression(SafeOneClassMixin, LogisticRegression):
        pass

    def fit(self, X, y):
        self.model = MultiOutputClassifier(self.SafeOneClassLogisticRegression()).fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)

