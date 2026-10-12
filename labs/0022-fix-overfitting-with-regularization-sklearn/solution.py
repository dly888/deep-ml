
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: LinearRegression() WILL overfit here.
    - LinearRegression(): Train R² ≈ 1.0, Val R² ≈ -5.0
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
    # TODO: fit a regularized model and return a predict function
    # 
    # Ideas:
    #   - Ridge(alpha=??) or RidgeCV (auto-tunes alpha)
    #   - Lasso(alpha=??) or LassoCV (auto-tunes + feature selection)
    #   - ElasticNet (combines L1 + L2)
    #   - SelectKBest + Ridge pipeline
    #   - Any sklearn approach that handles overfitting!
    #
    # Plain LinearRegression() will NOT work here.

    variances = np.std(X_train, axis=0)
    percentiles = np.searchsorted(
        np.sort(variances), variances, side="right"
    ) / len(variances) * 100

    idx = percentiles > 5

    X_train = X_train[:, idx]
    X_val = X_val[:, idx]

    alphas = [0.01, 0.1, 1, 10, 100, 1000]

    best_loss = np.inf
    best_model = None

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        loss = np.mean((y_val - y_pred) ** 2)

        if loss < best_loss:
            best_loss = loss
            best_model = model

    def predict(X):
        return best_model.predict(X[:, idx])

    return predict
