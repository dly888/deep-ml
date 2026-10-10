import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    n = len(X)

    X = X - np.mean(X, axis=0)
    sigma = (1 / (n - 1)) * (X.T @ X)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]

    variance_ratios = eigenvalues / np.sum(eigenvalues)

    return variance_ratios.tolist()


