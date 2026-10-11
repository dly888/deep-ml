import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    """
    Generate synthetic samples using SMOTE algorithm.

    Note: the random seed is set by the grader before your function runs,
    so you do NOT need to set it. Just use numpy's global RNG directly
    (np.random.randint, np.random.random, ...).

    Args:
        X_minority: 2D array of minority class samples (n_samples, n_features)
        n_synthetic: Number of synthetic samples to generate
        k: Number of nearest neighbors to consider

    Returns:
        2D array of synthetic samples (n_synthetic, n_features)
    """
    n_minority, n_features = X_minority.shape
    k = min(k, n_minority - 1)

    dists = np.linalg.norm(
        X_minority[:, None, :] - X_minority[None, :, :],
        axis=2
    )

    np.fill_diagonal(dists, np.inf)
    neighbours = np.argsort(dists, axis=1)[:, :k]

    synthetic = np.zeros((n_synthetic, n_features))

    for i in range(n_synthetic):
        base_idx = np.random.randint(0, n_minority)
        neighbour_idx = np.random.randint(0, k)
        gap = np.random.random()

        x = X_minority[base_idx]
        neighbour = X_minority[neighbours[base_idx, neighbour_idx]]

        synthetic[i] = x + gap * (neighbour - x)

    return synthetic

    