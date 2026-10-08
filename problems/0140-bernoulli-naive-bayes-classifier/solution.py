import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        n = len(y)
        d = len(X[0])

        self.prior = np.zeros(2)
        self.p = np.zeros((2, d))

        for c in range(2):
            mask = (y == c)
            X_c = X[mask]
            n_c = len(X_c)

            self.prior[c] = n_c / n

            for j in range(d):
                count = np.sum(X_c[:, j])
                self.p[c, j] = (count + self.smoothing) / (n_c + 2 * self.smoothing)

    def predict(self, X):
        # Predict class labels for test set X
        scores = np.zeros((len(X), 2))

        for c in range(2):
            if self.prior[c] == 0:
                scores[:, c] = -np.inf
                continue

            scores[:, c] = np.log(self.prior[c])

            for j in range(X.shape[1]):
                p = self.p[c, j]
                scores[:, c] += X[:, j] * np.log(p) + (1 - X[:, j]) * np.log(1 - p)

        return np.argmax(scores, axis=1)