from typing import Any

from matplotlib.axes import Axes
from pandas import DataFrame
from sklearn.datasets import make_blobs
import numpy as np


class Graph:

    algorithm_class: Any
    colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
    name: str

    centers = 1
    cluster_std = 1
    clicks = 1000

    @property
    def name(self):
        raise NotImplementedError

    def train_algorithm(self):
        raise NotImplementedError

    def __init__(self, ax: Axes) -> None:
        self.data = DataFrame()
        self.ax = ax
        self.ax.set_title(self.name)
        self.algorithm = self.algorithm_class()
        super().__init__()

    def render(self):
        grouped = self.data.groupby('cl')
        self.ax.clear()
        for key, group in grouped:
            group.plot(ax=self.ax, kind='scatter', x='x', y='y', label=key, color=self.colors[key])

    def generate_random_data(self):
        X, y = make_blobs(n_samples=self.clicks, centers=self.centers, n_features=2,
                          cluster_std=self.cluster_std)
        self.data = DataFrame(dict(x=X[:, 0], y=X[:, 1], cl=y))

    def add_click(self, x: int, y: int):
        prediction = self.algorithm.predict(np.asarray([[x, y]]))
        # Algorithms usually return a list of values for a list of provided data, but K Means just
        # classifies and returns integer straight away
        if isinstance(prediction, np.ndarray):
            predicted_cl = int(prediction[0][0])
        else:
            predicted_cl = prediction
        self.data = self.data.append(
            dict(x=x, y=y, cl=predicted_cl), ignore_index=True
        )
        self.render()
