
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
    tp, tn, fp, fn = 0, 0, 0, 0

    for a, b in data:
        if a == 1 and b == 1:
            tp += 1
        elif a == 0 and b == 0:
            tn += 1
        elif a == 0 and b == 1:
            fp += 1
        else:
            fn += 1

    return [[tp, fn], [fp, tn]]
