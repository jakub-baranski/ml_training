from time import sleep
from typing import Dict

from matplotlib.backend_bases import MouseEvent
from pandas import DataFrame
from sklearn.datasets import make_blobs
import numpy as np
from matplotlib import pyplot as plt
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


from algorithms.linear_regression import LinearRegression


@dataclass
class Dimensions:
    width: int
    height: int

    def size(self) -> tuple:
        return self.width, self.height


class Panel:

    axes = []
    clicks = []
    graphs_count = 5
    graphs = [
        {
            'name': 'Linear Regression',
            'centers': 1,
            'cluster_std': 1,
            'data': None,
            'ax': None
        },
        {
            'name': 'Logistic Regression',
            'centers': 2,
            'cluster_std': 0.5,
            'data': None,
            'ax': None
        }
    ]

    def __init__(self, width: int, height: int) -> None:
        self.dimensions = Dimensions(width=width, height=height)
        self._create_clicks()
        self.fig, axes = plt.subplots(len(self.graphs))
        for i, graph in enumerate(self.graphs):
            ax = axes[i]
            ax.set_title(graph['name'])
            graph['ax'] = axes[i]
        self._connect_mouse_listener()
        self._render_clicks()

    def show(self):
        plt.show()

    def _create_clicks(self):
        """
        Source: https://machinelearningmastery.com/generate-test-datasets-python-scikit-learn/
        """
        print('rolling blobs')
        for graph in self.graphs:
            X, y = make_blobs(n_samples=200, centers=graph['centers'], n_features=3,
                              cluster_std=graph['cluster_std'])
            graph['data'] = DataFrame(dict(x=X[:, 0], y=X[:, 1], cl=y))
            # scatter plot, dots colored by class value
            self.clicks.append(DataFrame(dict(x=X[:, 0], y=X[:, 1], cl=y)))

    def _render_clicks(self):
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
        for graph in self.graphs:
            grouped = graph['data'].groupby('cl')
            for key, group in grouped:
                group.plot(ax=graph['ax'], kind='scatter', x='x', y='y', label=key, color=colors[key])
            if graph['name'] == 'Linear Regression':
                self.draw_linear_regression(graph)

        plt.draw()

    def _connect_mouse_listener(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event: MouseEvent):
        """
        On click add new data to the clicks set that matches clicked region.
        """
        try:
            clicked_graph = next(g for g in self.graphs if g['ax'] is event.inaxes)
            clicked_graph['data'] = clicked_graph['data'].append(
                dict(x=event.xdata, y=event.ydata, cl=3), ignore_index=True)
            self._clear_clicks()
            self._render_clicks()
        except StopIteration:
            return

    def draw_linear_regression(self, graph: Dict):
        """
        Linear regression requires additional line so here we are...
        """
        data = graph['data']
        X = data.x
        y = data.y
        # Covert to numpy for algorithm
        X_train, X_test, y_train, y_test = map(lambda e: e.to_numpy(), train_test_split(X, y))

        algorithm = LinearRegression()

        algorithm.train_gradient_descent(X_train, y_train)

        y_p_test = algorithm.predict(X_test)

        algorithm.weights
        graph['ax'].plot(X_test, y_p_test)


    def reroll_initial_clicks(self):
        self._clear_clicks()
        self._create_clicks()
        self._render_clicks()

    def _clear_clicks(self):
        for graph in self.graphs:
            graph['ax'].cla()
