import numpy as np

def r_squared(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    
    mean_true = np.mean(y_true)
    sst = np.sum((y_true - mean_true) ** 2)
    
    r_squared = 1 - (ssr / sst)

    return r_squared