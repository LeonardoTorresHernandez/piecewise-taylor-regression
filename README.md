Piecewise Taylor Regression fits multiple local Taylor polynomials to data for flexible, accurate modeling of nonlinear functions.

![Visualization of Piecewise Taylor Regression](visualization.png)


# Piecewise Taylor Regression

Piecewise Taylor Regression is a Python package for fitting Taylor polynomial regressions to data, with the option to fit multiple Taylor polynomials at specified points (knots) for piecewise approximation. This approach allows for flexible, local polynomial modeling of complex, nonlinear data.

## What is Taylor Regression?

Taylor Regression fits a polynomial (Taylor series) to data around a specified center point. The Taylor polynomial approximates a function as a sum of powers of \((x-c)^n\), where \(c\) is the center. This is useful for locally approximating nonlinear functions with polynomials whose coefficients are determined by least squares regression.

## What is Piecewise Taylor Regression?

Piecewise Taylor Regression divides the input domain into segments, and fits a separate Taylor polynomial to each segment, centered at user-specified knots. This allows for accurate, flexible modeling of functions that change behavior in different regions, by combining several local Taylor approximations.

---

## Core Concept

**Taylor Regression** fits a polynomial (Taylor series) to data around a specified center point, approximating a function as a sum of powers of \((x-c)^n\), where \(c\) is the center. This is useful for locally approximating nonlinear functions with polynomials whose coefficients are determined by least squares regression.

**Piecewise Taylor Regression** divides the input domain into segments and fits a separate Taylor polynomial to each segment, centered at user-specified knots. This enables accurate, flexible modeling of functions that change behavior in different regions by combining several local Taylor approximations.

---

## Installation

Clone the repository and install in editable mode (recommended for development):

```bash
# Clone the repository
 git clone https://github.com/LeonardoTorresHernandez/piecewise-taylor-regression.git
 cd PiecewiseTaylorRegression

# Install dependencies and the package in editable mode
pip install -e .
```

Or, install dependencies directly:

```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## Quick Start

### Installation

Install the package and its dependencies:

```bash
pip install .
```

---

### TaylorRegression: Fit a single Taylor polynomial

```python
import numpy as np
from taylor_regression.core import TaylorRegression

# Generate synthetic data: y = 2 + 3x + 0.5x^2 + noise
np.random.seed(0)
X = np.linspace(-5, 5, 100)
y = 2 + 3*X + 0.5*X**2 + np.random.normal(0, 1, size=X.shape)

# Fit Taylor regression (degree=2) centered at 0
model = TaylorRegression(degree=2, center=0.0)
model.fit(X, y)

# Predict for a new value
x_new = 1.5
prediction = model.predict(x_new)
print(f"Prediction at x={x_new}: {prediction:.3f}")
```

### PiecewiseTaylorRegression: Fit multiple Taylor polynomials at specified knots

```python
import numpy as np
from taylor_regression.core import PiecewiseTaylorRegression

# Generate piecewise data: y = x^2 for x < 0, y = 2x + 1 for x >= 0, with noise
np.random.seed(1)
X = np.linspace(-5, 5, 200)
y = np.where(X < 0, X**2, 2*X + 1) + np.random.normal(0, 10, size=X.shape)

# Define knots (centers for Taylor expansions)
knots = np.array([-2.0, 2.0])

# Fit piecewise Taylor regression (degree=2)
model = PiecewiseTaylorRegression(degree=2)
model.fit(X, y, knots)

# Predict for a new value
x_new = 1.5
prediction = model.predict(x_new)
print(f"Piecewise prediction at x={x_new}: {prediction:.3f}")
```

---

## Running Tests

To run the tests using pytest, execute:

```bash
pytest
```

---

## Contributing

Feel free to open issues or contribute to the project via pull requests. All contributions are welcome!

For significant changes, please discuss them with the maintainers first by opening an issue.

---

## License

This project is licensed under the MIT License.
