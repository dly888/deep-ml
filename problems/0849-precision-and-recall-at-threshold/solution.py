import numpy as np

def precision_recall_at_threshold(y_true, y_scores, threshold):
    """
    Compute precision and recall at a given decision threshold.

    Args:
        y_true: list/array of true binary labels (0 or 1)
        y_scores: list/array of predicted scores in [0, 1]
        threshold: float, classification threshold (predict positive if score >= threshold)

    Returns:
        [precision, recall] as a list of two floats rounded to 4 decimals.
    """
    # Your code here
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    y_pred = y_scores >= threshold

    tp = np.sum((y_true == 1) & y_pred)
    fp = np.sum((y_true == 0) & y_pred)
    fn = np.sum((y_true == 1) & ~y_pred)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return [round(precision, 4), round(recall, 4)]
