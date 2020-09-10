from algorithms.linear_regression import LinearRegression
from graphs.graph import Graph

import numpy as np


class LinearRegressionGraph(Graph):

    algorithm_class = LinearRegression
    name = 'LinearRegression'
    clicks = 100

    def train_algorithm(self) -> None:
        X, y = (e.to_numpy() for e in [self.data.x, self.data.y])
        self.algorithm.train(X, y, n_iters=200, learning_rate=0.009)

    def render(self):
        super().render()
        self.draw_linear_regression()

    def add_click(self, x: int, y: int):
        """
        Since this is not a classification algorithm, I am overriding the default behaviour,
        where class in predicted and just add a point with class '3'.
        Then retrain algorithm and display new linear regression line.
        """
        self.data = self.data.append(
            dict(x=x, y=y, cl=3), ignore_index=True
        )
        self.train_algorithm()
        self.render()

    def draw_linear_regression(self):
        x = np.linspace(*self.ax.get_xlim(), 100)
        y_p = self.algorithm.predict(x)
        self.ax.plot(x, y_p)
