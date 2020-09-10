import numpy as np

from algorithms.logistic_regression import LogisticRegression
from graphs.graph import Graph


class LogisticRegressionGraph(Graph):

    algorithm_class = LogisticRegression

    name = 'Logistic Regression'
    centers = 2
    cluster_std = 0.5

    def train_algorithm(self) -> None:
        X = np.concatenate([self.data.x[:, np.newaxis], self.data.y[:, np.newaxis]], axis=1)
        y = self.data.cl.to_numpy()[:, np.newaxis]
        self.algorithm.train(X, y, n_iters=200, learning_rate=0.009)
