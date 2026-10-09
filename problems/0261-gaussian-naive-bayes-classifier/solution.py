
import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.

    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)

    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    n_train = len(X_train)
    D = len(X_train[0])
    classes = np.unique(y_train)
    n_classes = len(classes)

    priors = np.zeros(n_classes)
    means = np.zeros((n_classes, D))
    variances = np.zeros((n_classes, D))

    for i, c in enumerate(classes):
        mask = (y_train == c)
        X_c = X_train[mask]

        priors[i] = len(X_c) / n_train
        means[i] = np.mean(X_c, axis=0)
        variances[i] = np.var(X_c, axis=0)
	
    log_likelihood = -0.5 * np.sum(
        np.log(2 * np.pi * variances)
        + (X_test[:, None, :] - means)**2 / variances,
        axis=2
    )

    scores = log_likelihood + np.log(priors)

    return classes[np.argmax(scores, axis=1)]
