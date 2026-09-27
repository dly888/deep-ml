import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	X = np.array(X)
	y = np.array(y)
	XTX = X.T @ X
	XTY = X.T @ y 
	theta = np.linalg.inv(XTX) @ XTY
	theta = np.round(theta, decimals=4)
	return theta