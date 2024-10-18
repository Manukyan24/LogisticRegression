import numpy as np


class LogisticRegression:
    """
    Logistic regression implementation using numpy
    -----
    X must have (N_samples, N_features) dimension

    Y must have (N_samples, 1) dimension

    Weights vector initializes automatically and have shape (N_features, 1)
    """

    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_predict = self.sigmoid(z)
        return np.array([1 if i > 0.5 else 0 for i in y_predict]).reshape(-1, 1)

    def calc_acc(self, X, y):
        pred = self.predict(X)
        t = pred == y
        return np.sum(t.astype(int)) / len(t)

    def fit(self, X, y, X_val, y_val):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for epoch in range(self.epochs):
            A = self.sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / n_samples) * np.dot(X.T, A - y)
            db = (1 / n_samples) * np.sum(A - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            print(
                f"Epoch {epoch}/{self.epochs} | Train acc {self.calc_acc(X,y)*100 :.2f}% | Val acc {self.calc_acc(X_val,y_val)*100 :.2f}%"
            )
