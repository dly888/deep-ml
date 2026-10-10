import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    successes = np.sum(observations)
    failures = len(observations) - successes

    alpha_post = alpha + successes
    beta_post = beta + failures

    if alpha_post == 1 and beta_post == 1:
        return 0.5
    if alpha_post == 1:
        return 0.0
    if beta_post == 1:
        return 1.0

    return round(float((alpha_post - 1) / (alpha_post + beta_post - 2)), 4)