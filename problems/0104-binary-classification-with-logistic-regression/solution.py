import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	# Your code here
	def sigmoid(z):
		z = np.clip(z, -500, 500)
		return 1 / (1 + np.exp(-z))
	
	threshold = 0.5

	z = X @ weights + bias
	res = np.where(sigmoid(z) >= 0.5, 1, 0)
	return res.tolist()

