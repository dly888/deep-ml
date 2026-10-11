import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    thresholds = np.sort(np.unique(y_scores))[::-1]

    mask = y_scores[None, :] >= thresholds[:, None]

    tp = np.sum(mask & (y_true == 1), axis=1)
    fp = np.sum(mask & (y_true == 0), axis=1)
    fn = np.sum(~mask & (y_true == 1), axis=1)

    precision = np.where(tp + fp == 0, 1.0, tp / (tp + fp))
    recall = np.where(tp + fn == 0, 0.0, tp / (tp + fn))

    return (
    np.round(precision, 4).tolist(),
    np.round(recall, 4).tolist(),
    thresholds.tolist()
    )
