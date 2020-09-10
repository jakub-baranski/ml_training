from functools import partial

from algorithms.k_means import KMeans
from graphs.graph import Graph

import numpy as np


class KMeansGraph(Graph):

    name = 'KMeans'
    centers = 4
    algorithm_class = partial(KMeans, centers)

    def train_algorithm(self):
        X = np.concatenate([self.data.x[:, np.newaxis], self.data.y[:, np.newaxis]], axis=1)
        self.algorithm.fit(X)
