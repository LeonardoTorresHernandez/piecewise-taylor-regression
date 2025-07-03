"""
Test suite for PiecewiseTaylorRegression class in taylor_regression.core
- Fits synthetic piecewise polynomial data
- Visualizes fit
- Asserts low loss
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from taylor_regression.core import PiecewiseTaylorRegression


def test_piecewise_taylor_regression_fit_and_predict():
    """
    Test PiecewiseTaylorRegression on synthetic piecewise data.
    - y = x^2 for x < 0, y = 2x + 1 for x >= 0, with noise
    - Fits PiecewiseTaylorRegression (degree=2) with knots at -2, 2
    - Plots data and fit
    - Asserts loss is small
    """
    np.random.seed(1)
    X = np.linspace(-5, 5, 200)
    y_true = np.where(X < 0, X**2, 2 * X + 1)
    noise = np.random.normal(0, 10, size=X.shape)
    y = y_true + noise

    # Fit PiecewiseTaylorRegression with knots at -2, 2
    knots = np.array([-2.0, 2.0])
    model = PiecewiseTaylorRegression(degree=2)
    model.fit(X, y, knots)

    # Predict on X
    y_pred = np.array([model.predict(xi) for xi in X])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Noisy Data", alpha=0.5)
    plt.plot(X, y_true, label="True Function", color="green", linewidth=2)
    plt.plot(X, y_pred, label="Piecewise Taylor Fit", color="red", linestyle="--")
    plt.title("PiecewiseTaylorRegression Fit to Piecewise Data")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()

    # Show the plot interactively (optional, for local runs)
    plt.show()

    # Assert loss is small (fit is good)
    loss = model.loss(X, y)
    assert loss < 400, f"Loss too high: {loss}"


test_piecewise_taylor_regression_fit_and_predict()
