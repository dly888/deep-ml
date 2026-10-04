import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    X_train = np.asarray(X_train).flatten()
    X_val = np.asarray(X_val).flatten()
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    X_train_poly = np.vander(X_train, N=degree + 1, increasing=True)
    X_val_poly = np.vander(X_val, N=degree + 1, increasing=True)

    train_errors = []
    val_errors = []

    for n in train_sizes:
        X_n = X_train_poly[:n]
        y_n = y_train[:n]

        w = np.linalg.pinv(X_n) @ y_n

        train_pred = X_n @ w
        val_pred = X_val_poly @ w

        train_mse = np.mean((train_pred - y_n) ** 2)
        val_mse = np.mean((val_pred - y_val) ** 2)

        train_errors.append(float(train_mse))
        val_errors.append(float(val_mse))

    final_train = train_errors[-1]
    final_val = val_errors[-1]

    if final_train > bias_threshold:
        diagnosis = "high_bias"
    elif final_val - final_train > variance_threshold:
        diagnosis = "high_variance"
    else:
        diagnosis = "good_fit"

    return {
        "train_errors": train_errors,
        "val_errors": val_errors,
        "diagnosis": diagnosis
    }