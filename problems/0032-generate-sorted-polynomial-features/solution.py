import numpy as np
from itertools import combinations_with_replacement

def polynomial_features(X, degree):
    X = np.asarray(X)
    features = [np.ones(X.shape[0])]

    for d in range(1, degree + 1):
        for cols in combinations_with_replacement(range(X.shape[1]), d):
            features.append(np.prod(X[:, cols], axis=1))

    return np.sort(np.column_stack(features), axis=1)