import numpy as np

def accuracy_score(y_true, y_pred):
	# Your code here
	return np.sum(y_pred == y_true) / len(y_true)