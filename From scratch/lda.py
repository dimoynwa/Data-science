# Linear Discriminant Analysis

# The Goal is Feature reduction. The goal is to project a dataset into lower-dimentional
# space with good class-separability.

# PCA vs LDA
# PCA: finding a component axis tha MAXIMIZE the VARIANCE
# LDA: Additionally interested in the axis that MAXIMIZE the SEPARATION between classes
# LDA: LDA is supervised learning, PCA is unsupervised learning

# Math:
# 1. Within-class scatter matrix: Sw = sum(Sc), Sc = sum((x - meanXc) * (x - meanXc)Transposed)
# 2. Between-class scatter matrix = Sb = sum(nc * (meanXc - meanX) * (meanXc - meanX)Transponsed)
# 3. Calculate eigenvalues and eigenvectors for (Sw ^ -1) * Sb

# Approach: 1. Calculate Sw, 2. Calculate Sc, 3. Calculate eigenvalues for (Sw ^ -1) * Sb 
#   4. Sort eigenvectors by eigenvalues in DECR order, 5. Choose first K eigenvectors (linear discriminant).
# Transform the original N dimentional data points into K dimentional (Projection with dot Product)

import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class LDA:
    def __init__(self, k) -> None:
        # Number of dimentions
        self.k = k
    
    def fit(self, X, y):
        n_features = X.shape[1]
        classes = np.unique(y)

        # Mean of all columns, mean all shape is (1, n_features)
        mean_all = np.mean(X, axis=0)

        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            # (n_feat, n_c) * (n_c * n_feat) = (n_features, n_features) matrix
            S_w += np.dot((X_c - mean_c).T, X_c - mean_c)
            # nc * (n_feat, 1) * (1, n_feat) = (n_features, n_features) matrix
            mean_diff = (mean_c - mean_all).reshape((1, n_features))
            S_b += X_c.shape[0] * np.dot(mean_diff.T, mean_diff)

        #(Sw ^ -1) * Sb
        # (n_feat, n_feat) * (n_feat, n_feat) = (n_feat, n_feat) Matrix
        dot = np.dot(np.linalg.inv(S_w), S_b)
        eigenvalues, eigenvectors = np.linalg.eig(dot)
        
        eigenvectors = eigenvectors.T    
        # Sort the eigenvectors in DECR order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]

        # Store only first N eigenvectors
        self.linear_discriminants = eigenvectors[:self.k]


    def transform(self, X):
        # Project the data
        # (n_samp, n_feat) * (n_feat, self.k) = (n_samp, self.k) matrix
        return np.dot(X, self.linear_discriminants.T)

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target

    lda = LDA(2)
    lda.fit(X, y)

    print(f'Linear discriminants: {lda.linear_discriminants}')

    projections = lda.transform(X)
    print(f'Projections shape: {projections.shape}')

    x_1 = projections[:,0]
    x_2 = projections[:,1]

    plt.scatter(x_1, x_2, c=y)
    plt.xlabel('Linear discriminant 1')
    plt.ylabel('Linear discriminant 2')
    plt.colorbar()

    plt.show()