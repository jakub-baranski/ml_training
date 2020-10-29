from algorithms.kNN import kNN
from graphs.graph import Graph
import numpy as np


class KNNGraph(Graph):

    name = 'K Nearest Neighbour'
    algorithm_class = kNN
    centers = 3

    def train_algorithm(self):
        X = np.concatenate([self.data.x[:, np.newaxis], self.data.y[:, np.newaxis]], axis=1)
        y = self.data.cl.to_numpy()[:, np.newaxis]
        self.algorithm.fit(X, y)
