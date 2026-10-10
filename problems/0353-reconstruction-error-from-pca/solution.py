import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    n = len(X)

    A = (X - np.mean(X, axis=0))

    sigma = (1 / (n - 1)) * (A.T @ A)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)

    idx = np.argsort(eigenvalues)[::-1]
    top_n = idx[:n_components]

    eigenvectors = eigenvectors[:, top_n]

    for i in range(n_components):
        col = eigenvectors[:, i]
        first = np.flatnonzero(np.abs(col) > 0)

        if len(first) > 0 and col[first[0]] < 0:
            eigenvectors[:, i] *= -1
    
    Z = A @ eigenvectors
    X_hat_centered = Z @ eigenvectors.T
    X_hat = X_hat_centered + np.mean(X, axis=0)

    res = np.mean((X - X_hat) ** 2)

    return float(res)