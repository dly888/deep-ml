def balance_undersample(data: list) -> list:
    """
    Undersample the majority classes so all classes have the same number of
    samples equal to the minority class count.

    data: list of (sample, label) tuples
    Returns: list of (sample, label) tuples, order-preserving
    """
    if len(data) == 0:
        return []

    classes = {}

    for sample, label in data:
        if label not in classes:
            classes[label] = 1
        else:
            classes[label] += 1

    min_count = min(classes.values())

    keep = {}
    result = []

    for sample, label in data:
        if label not in keep:
            keep[label] = 0

        if keep[label] < min_count:
            result.append((sample, label))
            keep[label] += 1

    return result

