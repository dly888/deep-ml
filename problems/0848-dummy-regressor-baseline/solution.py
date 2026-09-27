import numpy as np

def dummy_regressor(y_train, n_test, strategy='mean', constant=None, quantile=None):
    """
    Baseline regressor that predicts a constant value derived from y_train.

    Args:
        y_train: 1D array-like of training target values.
        n_test: number of test predictions to return (int >= 0).
        strategy: one of 'mean', 'median', 'quantile', 'constant'.
        constant: required when strategy='constant'.
        quantile: required when strategy='quantile', must be in [0, 1].

    Returns:
        List[float] of length n_test, all equal to the chosen summary value.
    """
    if strategy == "mean":
        return [np.mean(y_train)] * n_test

    if strategy == "median":
        return [np.median(y_train)] * n_test

    if strategy == "quantile":
        if quantile is None:
            raise ValueError("Invalid input for quantile")
        elif quantile < 0 or quantile > 1:
             raise ValueError("Invalid input for quantile")
        else:
            return [np.quantile(y_train, quantile)] * n_test
    
    if strategy == "constant":
        return [constant] * n_test
    
    raise ValueError("Invalid strategy.")


