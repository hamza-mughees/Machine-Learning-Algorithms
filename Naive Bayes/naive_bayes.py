import numpy as np

class NaiveBayes:
    def __init__(self):
        self._classes = None
        self._var = None
        self._mean = None
        self._priors = None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self._classes = list(set(y))
        n_classes = len(self._classes)

        self._var = np.zeros((n_classes, n_features))
        self._mean = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for cl in self._classes:
            X_curr_cl = X[y == cl]
            self._var[cl, :] = np.var(X_curr_cl, axis=0)
            self._mean[cl, :] = np.mean(X_curr_cl, axis=0)
            self._priors[cl] = np.shape(X_curr_cl)[0] / n_samples

    def predict(self, X):
        return np.array([self._predict(p) for p in X])

    def _predict(self, p):
        posteriors = []

        for counter, curr_cl in enumerate(self._classes):
            prior = np.log(self._priors[counter])
            posteriors.append(prior + sum(np.log(self._gaussian_likelihood(p, counter))))
        
        return self._classes[np.argmax(posteriors)]

    def _gaussian_likelihood(self, p, cl_ind):
        numerator = np.exp(-((p-self._mean[cl_ind])**2)/2*(self._var[cl_ind]))
        denominator = (2*np.pi*self._var[cl_ind])**(1/2)
        return numerator/denominator
