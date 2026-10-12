import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: implement a training strategy that avoids overfitting
    # Some ideas:
    #   - Ridge regression (L2): add alpha * ||W||^2 to loss
    #   - Gradient descent with L2 penalty
    #   - Feature selection (remove low-variance or uncorrelated features)
    #   - Early stopping on gradient descent
    #   - Any combination of the above!
    variances = np.std(X_train, axis=0)
    percentiles = np.searchsorted(
        np.sort(variances), variances, side="right"
    ) / len(variances) * 100

    idx = percentiles > 95

    X_selected = X_train[:, idx]
    X_val = X_val[:, idx]

    alphas = [0.01, 0.1, 1, 10, 100, 1000]

    best_loss = np.inf
    best_weights = None
    best_bias = None

    x_mean = np.mean(X_selected, axis=0)
    y_mean = np.mean(y_train)

    X = X_selected - x_mean
    y = y_train - y_mean

    for alpha in alphas:
        weights = np.linalg.solve(
            X.T @ X + alpha * np.eye(X.shape[1]),
            X.T @ y
        )

        bias = y_mean - x_mean @ weights

        y_pred = X_val @ weights + bias
        loss = np.mean((y_val - y_pred) ** 2)

        if loss < best_loss:
            best_loss = loss
            best_weights = weights
            best_bias = bias

    def predict(X):
        return X[:, idx] @ best_weights + best_bias

    return predict  



