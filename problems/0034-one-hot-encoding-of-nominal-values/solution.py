import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	n = len(x)

	if n_col is None:
		n_col = np.max(x) + 1

	res = np.zeros(shape=(n, n_col))
	res[np.arange(n), x] = 1

	return res