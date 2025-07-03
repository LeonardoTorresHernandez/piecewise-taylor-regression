import numpy as np


class TaylorRegression:
    """
    TaylorRegression fits a polynomial (Taylor series) to data around a specified center.

    Parameters:
        degree (int): Degree of the Taylor polynomial.
        center (float): The point about which the Taylor expansion is centered.

    Example:
        >>> model = TaylorRegression(degree=2, center=0.0)
        >>> coeff = model.fit(X, y)
        >>> y_pred = model.predict(1.5)
    """

    def __init__(self, degree: int, center: float) -> None:
        """
        Initialize TaylorRegression.

        Args:
            degree (int): Degree of the Taylor polynomial.
            center (float): Center point for the Taylor expansion.
        """
        self.degree: int = degree
        self.center: float = center
        self.coeff: np.ndarray = np.ones(degree + 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit the Taylor polynomial to data (X, y).

        Args:
            X (np.ndarray): 1D array of input values.
            y (np.ndarray): 1D array of target values.

        Returns:
            np.ndarray: Array of fitted coefficients (constant first).

        Example:
            >>> model.fit(X, y)
        """
        # The goal in the fit function is to get the coefficients.
        # The number of coefficients depends on the deggree of the model to fit.
        # The coeffcients can be stored in a list.
        # Linear regression using (x^tx)^-1 X^TY will be applied to then using numpy

        # We can break this down into several steps.
        # Define the array for the coefficients.
        # Build the matrix for the multivariable linear regression using (x-c)^n
        # Calculate the formula and store the coefficients
        # Return the coefficients.

        # 1. Define the array for the coefficients.

        c = self.center
        degree = self.degree

        X = np.vander((X - c), N=degree + 1, increasing=True)

        # 2. Calculate the formula, we recall is (X^TX)^{-1}X^Ty
        # 2.1 Calculate X^T
        X_T = np.transpose(X)
        # 2.2 Calculate the rest of the formula using substitution u = X^T => (uX)^{-1}uy
        Coeff = np.linalg.inv(X_T @ X) @ (X_T @ y)

        # 3. Return coefficients and store.
        self.coeff = Coeff
        return Coeff

    def predict(self, value: float) -> float:
        """
        Predict the output for a given input value using the fitted Taylor polynomial.

        Args:
            value (float): Input value to predict.

        Returns:
            float: Predicted output.

        Example:
            >>> model.predict(1.5)
        """
        # Vectorized approach for better efficiency.
        # The formula for the Taylor regression is a_n(x-c)^n
        Terms = np.array(
            [self.coeff[i] * (value - self.center) ** i for i in range(self.degree + 1)]
        )
        prediction = np.sum(Terms)
        return prediction

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the sum of squared residuals (loss) for predictions on X compared to y.

        Args:
            X (np.ndarray): 1D array of input values.
            y (np.ndarray): 1D array of target values.

        Returns:
            float: Sum of squared residuals.

        Example:
            >>> model.loss(X, y)
        """
        SquareResiduals = np.array(
            [(y[i] - self.predict(X[i])) ** 2 for i in range(len(X))]
        )
        SumSquareRes = np.sum(SquareResiduals)
        return SumSquareRes


class PiecewiseTaylorRegression:
    """
    PiecewiseTaylorRegression fits multiple Taylor polynomials at specified knots
    for piecewise approximation.

    Parameters:
        degree (int): Degree of each Taylor polynomial.

    Example:
        >>> model = PiecewiseTaylorRegression(degree=2)
        >>> params = model.fit(X, y, knots=np.array([-2, 0, 2]))
        >>> y_pred = model.predict(1.5)
    """

    def __init__(self, degree: int):
        """
        Initialize PiecewiseTaylorRegression.

        Args:
            degree (int): Degree of each Taylor polynomial.
        """
        self.degree = degree

    def fit(self, X: np.ndarray, y: np.ndarray, knots: np.ndarray) -> np.ndarray:
        """
        Fit Taylor polynomials at each knot using the data.

        Args:
            X (np.ndarray): 1D array of input values.
            y (np.ndarray): 1D array of target values.
            knots (np.ndarray): 1D array of knot locations (centers for Taylor expansions).

        Returns:
            np.ndarray: 2D array where each column contains
                [center, coeff_0, ..., coeff_n] for a knot.

        Example:
            >>> model.fit(X, y, knots=np.array([-2, 0, 2]))
        """
        # The piecewise taylor regression uses several, smaller Taylor regression
        # models, the fit must fit a set amount of Taylor Regression models
        # The data is split into several "knots" which can be either selected
        # manually or automatically.
        # At each knot, a Taylor regression model will be fit and its
        # coefficients will be stored along with its knot index
        # After fitting all the models, the result will be stored in a matrix
        # the coefficients of each model as columns.
        # 1 Define the knots
        Params = []
        for center in knots:
            # Train a single TaylorRegression model
            model = TaylorRegression(self.degree, center)
            # Dynamic range adjustment
            initial_range = (X.max() - X.min()) / len(knots)
            max_range = X.max() - X.min()
            current_range = initial_range
            found = False
            while current_range <= max_range:
                Distances = np.abs(X - center)
                Selec_ind = np.where(Distances < current_range)[0]
                X_selec = X[Selec_ind]
                Y_selec = y[Selec_ind]
                # Check for enough unique points
                if len(np.unique(X_selec)) >= self.degree + 1:
                    found = True
                    break
                current_range *= 2  # Expand range
            if not found:
                # As a fallback, use all data
                X_selec = X
                Y_selec = y
            Coeff = model.fit(X_selec, Y_selec)
            MoData = np.concatenate([[center], Coeff])
            Params.append(MoData)
        Params = np.column_stack(Params)
        self.params = Params
        return Params

    def predict(self, x: float) -> float:
        """
        Predict the output for a given input value using the closest Taylor polynomial.

        Args:
            x (float): Input value to predict.

        Returns:
            float: Predicted output.

        Example:
            >>> model.predict(1.5)
        """
        # Get centers of the model's params
        Centers = self.params[0, :]
        Distances = np.abs(Centers - x)
        Closest = Distances.argmin()
        Center = Centers[Closest]

        # Predict the taylor series for that center
        Coeff = self.params[1:, Closest]
        # Evaluate with the center and the coefficients.
        Terms = np.array([Coeff[i] * (x - Center) ** i for i in range(self.degree + 1)])
        prediction = np.sum(Terms)
        return prediction

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the sum of squared residuals (loss) for piecewise predictions
        on X compared to y.

        Args:
            X (np.ndarray): 1D array of input values.
            y (np.ndarray): 1D array of target values.

        Returns:
            float: Sum of squared residuals.

        Example:
            >>> model.loss(X, y)
        """
        SquareResiduals = np.array(
            [(y[i] - self.predict(X[i])) ** 2 for i in range(len(X))]
        )
        SumSquareRes = np.sum(SquareResiduals)
        return SumSquareRes
