# Sigmoid function: s(x) = 1 / (1 + e^(-x))
# y_pred = hT(x) = 1 / (1 + e^(-wx + b))

# Calculating error - Crossentropy
# J(w, b) = J(hT) = 1/N * sum(yi * log(hT(xi)) + (1 - yi) * (log(1 - hT(xi))))

# Gradients:
# dJ/dw = 1/N * sum(2 * xi * (y_pred - y))
# dJ/db = 1/N * sum(2 * (y_pred - y))

# Steps:
# Training:
#   1. Initialize weights as 0
#   2. Initialize bias as 0
# Given a data pont
#   Prdict result using y = 1 / ( 1 + e^(-wx + b))
#   Calculate error (Crossentropy)
#   Use gradient descent to figure out new weights and bias
#   Repeat N times

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_pred, y):
    return np.sum(y_pred == y) / len(y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=100) -> None:
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((1, n_features)) # 1xfeatures matrix
        self.bias = 0

        for i in range(self.n_iters):
            # m = number of features, n = number of samples
            linear_pred = np.dot(self.weights, X.T) + self.bias # (1 x m) @ (m x n) + constant = (1 x n) matrix 
            y_pred = sigmoid(linear_pred)

            dw = (1 / n_samples) * np.dot((y_pred - y), X) * 2 # (1 x n) @ (n x m) = (1 x m) matrix
            db = (1 / n_samples) * np.sum((y_pred - y) * 2) 

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db


    def predict(self, X):
        linear_pred = np.dot(self.weights, X.T) + self.bias 
        y_pred = sigmoid(linear_pred)
        return [0 if y <= 0.5 else 1 for y in y_pred[0]]
    

if __name__ == '__main__':
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

    log_reg = LogisticRegression(n_iters=200)
    log_reg.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    print(y_pred)
    acc = accuracy(y_pred, y_test)
    print(f'Accuracy: ', acc)