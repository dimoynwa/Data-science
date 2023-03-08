# Given data point
#   1. Calculate its distance from all other data points in the dataset
#   2. Get the closest K points
#   3. Regression: Get the average of the K closest points
#   4. Classification: Get the label of majority value

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def euclidean_dist(point1, point2):
    return np.linalg.norm(point1 - point2)

class KNN:
    def __init__(self, k=3, dist_func=euclidean_dist) -> None:
        self.k = k
        self.dist_func = dist_func

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    # Predict multiple examples
    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    # Predict a single example
    def _predict(self, x):
        distances = np.array([self.dist_func(x, p) for p in self.X_train])
        labels = self.y_train[np.argsort(distances)][:self.k]
        res = np.bincount(labels).argmax()
        print(f'Choose between {labels}, result: {res}')
        return res    

if __name__ == '__main__':
    knn = KNN(k=5)
    iris_data = datasets.load_iris()
    X, X_test, y, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.15, random_state=1234)

    plt.figure()
    plt.scatter(iris_data.data[:,2], iris_data.data[:,3], c=iris_data.target)
    plt.show()

    knn.fit(X, y)

    predictions = knn.predict(X_test)

    print(f'Predictions: {predictions}')
    print(f'Actual: {y_test}')

    acc = np.sum(predictions == y_test) / len(predictions)
    print('Accuracy: ', acc)