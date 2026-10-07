
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	y = np.asarray(y)
	n = len(y)
	vals, cnts = np.unique(y, return_counts=True)
	
	pis = (cnts / n) ** 2
	val = 1 - (np.sum(pis))

	return round(val,3)