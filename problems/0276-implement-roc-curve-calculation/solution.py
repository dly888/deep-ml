import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    thresholds = np.sort(np.unique(y_scores))[::-1]

    mask = y_scores[None, :] >= thresholds[:, None]

    tp = np.sum(mask & (y_true == 1), axis=1)
    fp = np.sum(mask & (y_true == 0), axis=1)
    fn = np.sum(~mask & (y_true == 1), axis=1)
    tn = np.sum(~mask & (y_true == 0), axis=1)

    tpr = np.where(tp + fn == 0, 0.0, tp / np.maximum(tp + fn, 1))
    fpr = np.where(fp + tn == 0, 0.0, fp / np.maximum(fp + tn, 1))

    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.insert(fpr, 0, 0.0)

    return (
        np.round(fpr, 4).tolist(),
        np.round(tpr, 4).tolist()
    )