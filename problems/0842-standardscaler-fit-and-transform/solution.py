import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    n = len(X_train)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    mask = (np.isclose(std, 0))
    std[np.isclose(std, 0)] = 1
    
    X_test = (X_test - mean) / std
    return X_test
    
