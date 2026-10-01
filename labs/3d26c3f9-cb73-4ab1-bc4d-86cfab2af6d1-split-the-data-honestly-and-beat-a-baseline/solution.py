import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    n = len(X)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)

    train_end = int(train_frac * n)
    val_end = train_end + int(val_frac * n)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    mean = np.mean(y_train)
    test_predictions = np.full(len(test_idx), mean)


    return test_predictions, train_idx, val_idx, test_idx



