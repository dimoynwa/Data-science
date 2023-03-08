# Principal Component Analysis
# PCA finds a new set of dimentions such that all the dimensions are ORTHOGONAL
# (and hence linearly independent) and ranked according to the variance of data along them.

# Find a transformation such that
# 1. The transformed features are linearly independent
# 2. Dimentinality can be reduced by taking only the dimentions with the highest importance
# 3. Newly found dimentions should minimize the projection error
# 4. The projected points should have maximum spread, i.e. maximum variance

# Variance -> Var(X) = 1/n * sum[(Xi - averageX)^2]
# Covariance Matrix -> Cov(X,Y) = 1/n * sum[((Xi - averageX) * (Yi - averageY))Transposed]

# Aproach:
# 1. Substract the mean from X
# 2. Calculate Cov(X,X)
# 3. Calculate eigenvectors and eigenvalues 
# 4. Sort the eigenvectors according to their eigenvalues in DECREASING order
# 5. Choose first K eigenvectors and that will be our new K dimentions
# 6. Transform the original n dimentional data into K dimentions (Projections with dot product)

import numpy as np

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def fit(self, X):
        # Calculate the Mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        # Calculate the Covariance matrix. 1 row = 1 sample, 1 column = 1 feature
        cov = np.cov(X.T)
        # Eigenvectors and Eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov) # Eigenvectors is returned as 1 column = 1 eigenvector, so we transpose it
        eigenvectors = eigenvectors.T    
        # Sort the eigenvectors in DECR order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]

        # Store only first N eigenvectors
        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # Project the data
        X = X - self.mean
        return np.dot(X, self.components.T)
    
if __name__ == '__main__':
    data = load_iris()
    X = data.data
    y = data.target

    # 2 principal components
    pca = PCA(2)
    pca.fit(X)

    X_projected = pca.transform(X)
    print(f'Shape of X: {X.shape}')
    print(f'Shape of X_projected: {X_projected.shape}')

    plt.scatter(X_projected[:,0], X_projected[:,1], c=y)
    plt.colorbar()
    plt.xlabel('Principal component 1')
    plt.ylabel('Principal component 2')
    plt.show()