import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    idx = np.argsort(y_scores)[::-1]
    thresholds = y_scores[idx]

    mask = y_scores[None, :] >= thresholds[:, None]

    tp = np.sum(mask & (y_true == 1), axis=1)
    fp = np.sum(mask & (y_true == 0), axis=1)
    fn = np.sum(~mask & (y_true == 1), axis=1)
    tn = np.sum(~mask & (y_true == 0), axis=1)

    tpr = np.where(tp + fn == 0, 0.0, tp / np.maximum(tp + fn, 1))
    fpr = np.where(fp + tn == 0, 0.0, fp / np.maximum(fp + tn, 1))

    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.insert(fpr, 0, 0.0)

    auc = 0.5 * np.sum((tpr[:-1] + tpr[1:]) * (fpr[1:] - fpr[:-1]))

    return float(auc)
    