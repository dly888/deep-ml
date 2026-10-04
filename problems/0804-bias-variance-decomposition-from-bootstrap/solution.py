import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    predictions = np.asarray(predictions)
    y_true = np.asarray(y_true)

    means = np.mean(predictions, axis=0)
    bias_squared = np.mean((means - y_true) ** 2)
    variance = np.mean((predictions - means) ** 2)
    mse = np.mean((predictions - y_true) ** 2)

    return {
        "bias_squared": float(bias_squared),
        "variance": float(variance),
        "mse": float(mse)
    }
