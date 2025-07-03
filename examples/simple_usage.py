import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from taylor_regression.core import TaylorRegression, PiecewiseTaylorRegression

# 1. Generate a noisy sine wave dataset
np.random.seed(42)
X = np.linspace(-3 * np.pi, 4 * np.pi, 200)
y_true = np.sin(X)
# Reduce noise for a cleaner fit
demo_noise_std = 0.2  # Lower noise for demo
noise = np.random.normal(0, demo_noise_std, size=X.shape)
y_noisy = y_true + noise

# Reshape for sklearn
X_2d = X.reshape(-1, 1)

# 2. Fit models
# 2.1 Standard LinearRegression
linreg = LinearRegression()
linreg.fit(X_2d, y_noisy)
y_pred_linreg = linreg.predict(X_2d)

# 2.2 TaylorRegression (degree=4, center at 0)
taylor = TaylorRegression(degree=4, center=0.0)
taylor.fit(X, y_noisy)
y_pred_taylor = np.array([taylor.predict(x) for x in X])

# 2.3 PiecewiseTaylorRegression (degree=4, n_segments=12)
n_segments = 12  # More segments for better fit
knots = np.linspace(X.min(), X.max(), n_segments)
piecewise = PiecewiseTaylorRegression(degree=4)  # Higher degree for better local fit
piecewise.fit(X, y_noisy, knots=knots)
y_pred_piecewise = np.array([piecewise.predict(x) for x in X])

# 3. Plot
plt.figure(figsize=(12, 6))
plt.scatter(X, y_noisy, color="gray", alpha=0.5, label="Noisy data")
plt.plot(X, y_true, color="black", linestyle="--", label="True sin(x)")
plt.plot(X, y_pred_linreg, color="blue", label="LinearRegression")
plt.plot(X, y_pred_taylor, color="green", label="TaylorRegression (deg=4)")
plt.plot(
    X,
    y_pred_piecewise,
    color="red",
    linewidth=3,
    label=f"PiecewiseTaylorRegression (deg={piecewise.degree}, {n_segments} segs)",
)
plt.title("Piecewise Taylor Regression vs. Linear and Taylor Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()
