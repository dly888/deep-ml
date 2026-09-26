def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    mn, mx = min(x), max(x)

    for i in range(len(x)):
        x[i] = (x[i] - mn) / (mx - mn)
    
    return x
    