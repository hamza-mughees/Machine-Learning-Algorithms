import numpy as np
from quicksort import Quick_Sort
from collections import Counter

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def _euclidean_distance(self, p, q):
        return np.sqrt(np.sum((q-p)**2))
    
    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return np.array([self._predict(p) for p in X])

    def _predict(self, p):
        distances = [self._euclidean_distance(p, q) for q in self.X]
        qs = Quick_Sort()
        qs.sort(distances)
        n_nearest_neighbors_indeces = qs.inds[:self.n_neighbors]
        n_nearest_neighbors_y = [self.y[i] for i in n_nearest_neighbors_indeces]
        return Counter(n_nearest_neighbors_y).most_common(1)[0][0]
