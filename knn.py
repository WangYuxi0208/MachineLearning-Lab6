class KNN:
    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X = X_train
        self.y = y_train

    def predict_one(self, x):
        dist = np.linalg.norm(self.X - x, axis=1)
        idx = dist.argsort()[:self.k]
        labels = self.y[idx]
        return np.bincount(labels).argmax()

    def predict(self, X_test):
        return np.array([self.predict_one(x) for x in X_test])
