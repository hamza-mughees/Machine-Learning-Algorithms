import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, max_iter=1000):
        self.intercept_ = None   # when all feature inputs are zero (w0)
        self.coef_ = None   # weights of each feature (w1x1 + w2x2 + ... + wnxn)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
    
    def _sigmoid(self, z): return 1/(1+np.exp(-z))

    def predict_proba(self, X):
        return self._sigmoid(self.intercept_ + np.dot(X, self.coef_))
    
    def predict(self, X):
        return np.array([1 if proba >= 0.5 else 0 for proba in self.predict_proba(X)])
    
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self.intercept_ = 0
        self.coef_ = np.zeros(n_features)

        for _ in range(self.max_iter):
            y_pred = self.predict_proba(X)

            D_intercept_ = (1/n_samples)*np.sum(y_pred-y)
            D_coef_ = (1/n_samples)*np.dot(X.T, (y_pred-y))

            self.intercept_ -= D_intercept_*self.learning_rate
            self.coef_ -= D_coef_*self.learning_rate
