import numpy as np

class SVC:
    def __init__(self, learning_rate=0.001, max_iters=1000, l=0.001):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.l = l
        self.intercept_ = None
        self.coef_ = None
    
    def fit(self, X, y):
        y_tmp = self._sign(y)
        n_samples, n_features = np.shape(X)

        self.intercept_ = 0
        self.coef_ = np.zeros(n_features)

        for _ in range(self.max_iters):
            for counter, curr_sample in enumerate(X):
                coef_derivative_offset = 0

                if y_tmp[counter]*(np.dot(curr_sample, self.coef_) - self.intercept_) < 1:
                    self.intercept_ -= y_tmp[counter]*self.learning_rate
                    coef_derivative_offset = np.dot(curr_sample, y_tmp[counter])
                
                self.coef_ -= ((2*self.l*self.coef_) - coef_derivative_offset)*self.learning_rate
    
    def predict(self, X):
        return self._unSign(np.dot(X, self.coef_) - self.intercept_)
    
    def _sign(self, y):
        return [-1 if y[i] <= 0 else 1 for i in range(len(y))]
    
    def _unSign(self, y):
        return np.array([0 if y[i] <= 0 else 1 for i in range(len(y))])
