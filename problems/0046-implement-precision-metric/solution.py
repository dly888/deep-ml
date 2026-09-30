import numpy as np
def precision(y_true, y_pred):
	# Your code here
	tp = np.sum(y_true & y_pred)
	fp = np.sum((y_pred == 1) & (y_true == 0))

    if tp + fp == 0:
        return 0.0

	return tp / (tp + fp)