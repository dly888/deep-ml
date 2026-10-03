import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	y_pred = X @ w
	mse = np.mean((y_true - y_pred) ** 2)
	penalty = alpha * np.sum(w ** 2)

	return mse + penalty
