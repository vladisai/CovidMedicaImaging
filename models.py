import numpy as np


class Baseline():

    def __init__(self):
        pass

    def fit(self, X):
        pass

    def predict(self, x):
        return np.zeros_like(x['lab'])
