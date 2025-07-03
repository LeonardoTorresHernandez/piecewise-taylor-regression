# Taylor series
The Taylor series is an infinite mathematical approach that converts any function $f(x)$ into a power series of the form:
$$
\sum_{n=0}^{\infty}a_{n}(x-c)^n
$$
Where $a_{n}$ is the coefficient for each term.
Where $c$ is the center of the series.
Where $x$ is the value at which the series is evaluated.

For any function $f(x)$, we can approximate it using a power series:
$$
f(x) ≈ \sum_{n=0}^{\infty} a_{n}(x-c)^n
$$
This approach is called the Taylor series. However, a problem arises, how do we find the coefficients that best approximate the power series to the function; After a thorough analysis from first principles into the series and the function, the following formula was derived to get the best coefficients for the function:
$$
a_{n}= \frac{f^{(n)}(c)}{n!}
$$

Where $f^{(n)}(c)$ is the n-th derivative of the function $f(x)$ evaluated at the center point $c$.
Where $n!$ is the factorial of $n$.
Where $a_{n}$ is the coefficient we are finding.
# OLS
The ordinary least squares method involves finding the best line to fit the data, it takes the form of:
$$
f(x)=ax+b
$$
$$
a = \frac{\sum x_{i}y_{i} - \bar{y} \sum x_{i}}{\sum x_{i}^2- \bar{ x}\sum x_{i}}
$$
$$
b = \bar{y} - a \bar{x}
$$



# Taylor regression
All these theory is great, but how do we map this mathematical series into real world data? That is where the Taylor series arises, it is a practical version of the Taylor series based in the same concepts but designed to be computationally possible and efficient. Unlike the Taylor series, the Taylor series is **not** an infinite series but is instead approximated to a predefined number of terms, increasing the accuracy as the terms increase and reducing it as the terms decrease; The Taylor regression defines a power series which for which the coefficients are solved using OLS. It has a fixed number of terms and takes the form of:
$$
f(x) = \sum_{n=1}^k \beta_{n} (x-c)^n
$$
Where the coefficients $\beta$ are solved using linear regression.
Where $x$ is the value at which the function is evaluated.
Where $c$ is the center.

# Piecewise Taylor regression
The accuracy of the Taylor regression is determined by two key variables:
- The center: As the value to evaluate approaches the center, the accuracy increases and as it recedes from the center, the accuracy increases.
- The number of terms: As the number of terms increases, so does the accuracy and when it decreases, so does the accuracy. However, more terms means more computationally expensive.

To counter the accuracy loss by center, we divide the data into several sections, foreach section, we fit a Taylor regression model with a high accuracy over that Area; To predict, we find the area the value to predict falls into and use the model of that area.


# Calculating the coefficients.

To calculate the coefficients, we will convert the dataset of (x,y) into a data set of:
$$
(x-c)^n
$$
With $k$ independent variables,. Then, a linear regression model will be fitted for each coefficient using the multiple variable OLS formula:
$$
(X^TX)^{-1}X^TY
$$
