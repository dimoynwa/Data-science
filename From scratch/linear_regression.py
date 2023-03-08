# Steps:
# Training:
#   1. Initialize weights as 0
#   2. Initialize bias as 0
# Given a data pont
#   Prdict result using y = wx + b
#   Calculate error (MSE - Mean squared error)
#   Use gradient descent to figure out new weights and bias
#   Repeat N times

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# Updating params 
# dJ/dw = 1/N * sum[i 1..N] (2*Xi*(Yi - yi))
# dJ/db = 1/N * sum[1 1..N] (2*(Yi - yi))

def mse(y_pred, y):
    return np.mean((y_pred - y)**2) 

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 100, err=mse) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.err = err
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((1, n_features)) # 1xfeatures matrix
        self.bias = 0

        for i in range(self.n_iters):
            # m = number of features, n = number of samples
            y_pred = np.dot(self.weights, X.T) + self.bias # (1 x m) @ (m x n) + constant = (1 x n) matrix 

            if i % 20 == 0:
                err = self.err(y_pred, y)
                print(f'Error on iteration {i} is {err}')

            dw = (1 / n_samples) * np.dot((y_pred - y), X) * 2 # (1 x n) @ (n x m) = (1 x m) matrix
            db = (1 / n_samples) * np.sum((y_pred - y) * 2) 

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        y_pred = np.dot(self.weights, X.T) + self.bias
        return y_pred
    
if __name__ == '__main__':
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

    lin_reg = LinearRegression(lr=0.01, n_iters=150)
    lin_reg.fit(X_train, y_train)

    predictions = lin_reg.predict(X_test)

    final_err = mse(predictions, y_test)
    print(f'Final err on test samples: {final_err}')

    final_pred = lin_reg.predict(X)

    plt.figure()
    m1 = plt.scatter(X_train, y_train, c='blue', marker='.')
    m1 = plt.scatter(X_test, y_test, c='red', marker='.')
    plt.plot(X, final_pred.T, c='black', linewidth=2)
    plt.show()
