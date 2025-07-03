"""
Test suite for TaylorRegression class in taylor_regression.core
- Fits synthetic polynomial data
- Visualizes fit
- Asserts low loss
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from taylor_regression.core import TaylorRegression


def test_taylor_regression_fit_and_predict():
    """
    Test TaylorRegression on synthetic quadratic data.
    - Generates y = 2 + 3x + 0.5x^2 + noise
    - Fits TaylorRegression (degree=2)
    - Plots data and fit
    - Asserts loss is small
    """
    np.random.seed(0)
    X = np.linspace(-5, 5, 100)
    y_true = 2 + 3 * X + 0.5 * X**2
    noise = np.random.normal(0, 1, size=X.shape)
    y = y_true + noise

    # Fit TaylorRegression centered at 0
    model = TaylorRegression(degree=2, center=0.0)
    coeff = model.fit(X, y)

    # Predict on X
    y_pred = np.array([model.predict(xi) for xi in X])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Noisy Data", alpha=0.5)
    plt.plot(X, y_true, label="True Function", color="green", linewidth=2)
    plt.plot(X, y_pred, label="Taylor Fit", color="red", linestyle="--")
    plt.title("TaylorRegression Fit to Quadratic Data")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Assert loss is small (fit is good)
    loss = model.loss(X, y)
    assert loss < 200, f"Loss too high: {loss}"

    # Assert coefficients are close to true values
    assert np.allclose(coeff, [2, 3, 0.5], atol=0.5), f"Coefficients off: {coeff}"


test_taylor_regression_fit_and_predict()
