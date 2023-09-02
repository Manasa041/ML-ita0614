import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Generating random data points
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
# Fit the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Predict using the Linear Regression model
y_pred_linear = lin_reg.predict(X)
# Transforming the input features to include polynomial terms
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Fit the Polynomial Regression model
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

# Predict using the Polynomial Regression model
y_pred_poly = poly_reg.predict(X_poly)
# Plot the original data
plt.scatter(X, y, label='Data Points')

# Plot the Linear Regression line
plt.plot(X, y_pred_linear, color='red', label='Linear Regression')

# Sort the X values for smoother Polynomial Regression line
X_sorted, y_pred_poly_sorted = zip(*sorted(zip(X, y_pred_poly)))
plt.plot(X_sorted, y_pred_poly_sorted, color='green', label='Polynomial Regression')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
