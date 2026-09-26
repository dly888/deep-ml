import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	n = len(data)
	mean = np.mean(data, axis=0)
	std = np.std(data, axis=0)
	mn, mx = np.min(data, axis=0), np.max(data, axis=0)

	standardised = (data - mean) / std
	minmaxed = (data - mn) / (mx - mn)

	return standardised, minmaxed