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
from algorithms.logistic_regression import LogisticRegression


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
            'ax': None,
            'algorithm': None,
            'algorithm_class': LinearRegression
        },
        {
            'name': 'Logistic Regression',
            'centers': 2,
            'cluster_std': 0.5,
            'data': None,
            'ax': None,
            'algorithm': None,
            'algorithm_class': LogisticRegression
        }
    ]

    # TODO: Make graph a separate class. Make algorithm part of this object. Train it once. Separate training (initial) data from newle created

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
            X, y = make_blobs(n_samples=1000, centers=graph['centers'], n_features=2,
                              cluster_std=graph['cluster_std'])
            graph['data'] = DataFrame(dict(x=X[:, 0], y=X[:, 1], cl=y))
            # scatter plot, dots colored by class value
            self.clicks.append(DataFrame(dict(x=X[:, 0], y=X[:, 1], cl=y)))

    def _render_clicks(self):
        for graph in self.graphs:
            self._render_graph(graph)

        plt.draw()

    def _render_graph(self, graph: Dict) -> None:
        colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
        grouped = graph['data'].groupby('cl')
        ax = graph.get('ax')
        ax.clear()
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
        if graph['name'] == 'Linear Regression':
            self.draw_linear_regression(graph)

    def _connect_mouse_listener(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event: MouseEvent):
        """
        On click add new data to the clicks set that matches clicked region.
        """
        try:
            clicked_graph = next(g for g in self.graphs if g['ax'] is event.inaxes)
            predicted_cl = 3

            if clicked_graph['name'] == 'Logistic Regression':
                algorithm = self._train_logistic_regression(clicked_graph)
                prediction = algorithm.predict([[event.xdata, event.ydata]])

                predicted_cl = prediction[0][0]

            clicked_graph['data'] = clicked_graph['data'].append(
                dict(x=event.xdata, y=event.ydata, cl=predicted_cl), ignore_index=True)

            self._render_graph(clicked_graph)
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

        algorithm.train(X_train, y_train)

        y_p_test = algorithm.predict(X_test)

        algorithm.weights
        graph['ax'].plot(X_test, y_p_test)

    def _train_logistic_regression(self, graph: Dict):
        # x and y on the graph are features, join them here
        data = graph['data']
        X = np.concatenate([data.x[:, np.newaxis], data.y[:, np.newaxis]], axis=1)

        y = data.cl.to_numpy()[:, np.newaxis]

        algorithm = LogisticRegression()
        algorithm.train(X, y, n_iters=200, learning_rate=0.009)
        return algorithm

    def _predict_logistic_regression(self, algorithm, X):
        algorithm.predict(X)


    def reroll_initial_clicks(self):
        self._clear_clicks()
        self._create_clicks()
        self._render_clicks()

    def _clear_clicks(self):
        for graph in self.graphs:
            graph['ax'].cla()
