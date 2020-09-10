from dataclasses import dataclass
from typing import List, Type

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent

from graphs.kNN import KNNGraph
from graphs.k_means import KMeansGraph
from graphs.logistic_regression import LogisticRegressionGraph
from graphs.linear_regression import LinearRegressionGraph
from graphs.graph import Graph


class Panel:

    graph_classes: List[Type[Graph]] = [
        LinearRegressionGraph,
        LogisticRegressionGraph,
        KNNGraph,
        KMeansGraph,
    ]

    graphs: List[Graph] = []

    def __init__(self) -> None:
        self.fig, axes = plt.subplots(len(self.graph_classes))
        self._create_graphs(axes)
        self._connect_mouse_listener()
        plt.draw()

    def show(self):
        plt.show()

    def on_click(self, event: MouseEvent):
        try:
            clicked_graph = next(g for g in self.graphs if g.ax is event.inaxes)
            clicked_graph.add_click(event.xdata, event.ydata)
            self.fig.canvas.draw()
        except StopIteration:
            return

    def _create_graphs(self, axes):
        for i, graph_cl in enumerate(self.graph_classes):
            graph = graph_cl(axes[i])
            graph.generate_random_data()
            graph.train_algorithm()
            graph.render()
            self.graphs.append(graph)

    def _connect_mouse_listener(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
