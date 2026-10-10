import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    n = len(data)

    X = (data - np.mean(data, axis=0)) / np.std(data, axis=0, ddof=1)

    sigma = (1 / (n - 1)) * (X.T @ X)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)

    idx = np.argsort(eigenvalues)[::-1]
    top_k = idx[:k]

    eigenvectors = eigenvectors[:, top_k]

    for i in range(k):
        col = eigenvectors[:, i]
        first = np.flatnonzero(np.abs(col) > 1e-10)

        if len(first) > 0 and col[first[0]] < 0:
            eigenvectors[:, i] *= -1

    return np.round(eigenvectors, 4)
