from dataclasses import dataclass
from typing import List, Type

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseEvent

from graphs.kNN import KNNGraph
from graphs.k_means import KMeansGraph
from graphs.logistic_regression import LogisticRegressionGraph
from graphs.linear_regression import LinearRegressionGraph
from graphs.graph import Graph


@dataclass
class Dimensions:
    width: int
    height: int

    def size(self) -> tuple:
        return self.width, self.height


class Panel:

    graph_classes: List[Type[Graph]] = [
        LinearRegressionGraph,
        LogisticRegressionGraph,
        KNNGraph,
        KMeansGraph,
    ]

    graphs: List[Graph] = []

    def __init__(self, width: int, height: int) -> None:
        self.dimensions = Dimensions(width=width, height=height)
        self.fig, axes = plt.subplots(len(self.graph_classes))

        for i, graph_cl in enumerate(self.graph_classes):
            graph = graph_cl(axes[i])
            graph.generate_random_data()
            graph.train_algorithm()
            graph.render()
            self.graphs.append(graph)

        self._connect_mouse_listener()
        plt.draw()

    def show(self):
        plt.show()

    def _connect_mouse_listener(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event: MouseEvent):
        """
        On click add new data to the clicks set that matches clicked region.
        """
        try:
            clicked_graph = next(g for g in self.graphs if g.ax is event.inaxes)
            clicked_graph.add_click(event.xdata, event.ydata)
            self.fig.canvas.draw()
        except StopIteration:
            return
