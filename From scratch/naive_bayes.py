# Probabilistic classifier based on Bayes theorem
# Strong (naive) independace assumptions between features
# Bayes theorem: P(A|B) = (P(B|A) * P(A)) / P(B)

# In our case: P(y|X) = (P(x1|y)*P(x2|y)*...*P(xn|y)*P(y))/P(y)
# P(X) not depends on y, so we can skip it
# And the final result will be: argmax_y P(x1|y)*P(x2|y)*...*P(xn|y)*P(y)
# As P(xi|y) are very small numbers between 0 and 1, we can have inaccuracies
# So we do a little trick: Instead of product we do SUM, and apply an log function
# y = argmax_y (log(P(x1|y)) + log(P(x2|y)) + ... + log(P(xn|y)) + log(P(y)))

# Prior and class conditional 
# P(y) -> frequency of each class
# P(xi|y) -> class conditional frequency = P(xi|y) = 1 / sqrt(2 * Pi * (sig(y)^2) ) * exp(- (xi - Mu_y)^2 / 2 * (sig(y)^2))
# sig(y) - Variance of y, Mu_y = mean value of y

# Steps:
# Training:
#   Calculate mean, var and prior(frequency) for each class
# Prediction:
#   Calculate posterior for each class 
#   Choose class with highest posterior 

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate mean, var and prior(frequency) for each class
        # For ench class we have a row and for each feature column
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        # For ench class we have a row and for each feature column
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        # 1 row with priors for each class
        self._prior = np.zeros(n_classes, dtype=np.float64)
        
        for idx, c in enumerate(self._classes):
            X_c = X[y==c]
            # Means for every feature in class with index idx
            self._mean[idx, :] = np.mean(X_c, axis=0)
            # Varianve for every feature in class with index idx
            self._var[idx, :] = np.var(X_c, axis=0)
            # Prior for class with index idx
            self._prior[idx] = X_c.shape[0] / float(n_samples)


    def predict(self, X):
        return [self._predict_one(x) for x in X]

    def _predict_one(self, x):
        predictions =[self._prob_dens(x, cls_idx)  for cls_idx, _ in enumerate(self._classes)]
        return self._classes[np.argmax(predictions)]

    
    def _prob_dens(self, x, cls_idx):
        # 1 x n_features vector
        mean = self._mean[cls_idx]
        var = self._var[cls_idx]

        # Calculate P(xi|y) = 1 / sqrt(2 * Pi * (sig(y)^2) ) * exp(- (xi - Mu_y)^2 / 2 * (sig(y)))
        num = np.exp(- (x - mean) ** 2 / (2 * var ))
        den = np.sqrt(2 * np.pi * var)

        return np.sum(np.log(num / den)) + np.log(self._prior[cls_idx])

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

if __name__ == '__main__':
    X, y =datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    naive_bayes = NaiveBayes()
    naive_bayes.fit(X_train, y_train)

    predictions = naive_bayes.predict(X_test)

    acc = accuracy(y_test, predictions)
    print(f'Accuracy: {acc}')