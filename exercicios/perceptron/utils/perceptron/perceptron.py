import numpy as np

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
class Perceptron:
    def __init__(
        self,
        x,
        y,
        weights=None,
        bias=None,
        eta: float = 0.01,
        att_bias: bool = False,
        max_epochs: int = 100
    ):
        self.x = x
        self.y = y
        
        n_features = x.shape[1]

        if weights is None:
            self.weights = np.random.random(n_features)
        else:
            self.weights = weights

        if bias is None:
            self.bias = np.random.random()
        else:
            self.bias = bias
        
        self.eta = eta
        self.att_bias = att_bias
        self.max_epochs = max_epochs

        self.accuracy_per_epoch = []
        self.updates_per_epoch = []
        self.converged = False
        self.epochs_run = 0

    def train(self):
        for epoch in range(self.max_epochs):
            updates = 0

            for x, y in zip(self.x, self.y):
                y_pred = 1 if (np.dot(x, self.weights) + self.bias) >= 0 else 0
                error = y - y_pred

                if error != 0:
                    updates += 1
                    self._rebalance(x=x, error=error)

            acc = 1.0 - (updates / len(self.y))
            self.accuracy_per_epoch.append(acc)
            self.updates_per_epoch.append(updates)

            self.epochs_run = epoch + 1
            if updates == 0:
                self.converged = True
                break

    def predict(self, X_test=None, y_test=None):
        if X_test is None:
            raise ValueError("Entre com o X_test")

        scores = X_test @ self.weights + self.bias
        y_pred = (scores >= 0).astype(int)

        if y_test is not None:
            acc = (y_pred == y_test).mean()
            return y_pred, acc
        return y_pred

    def _rebalance(self, x, error):
        if error != 0:
            self.weights += self.eta * error * x
            if self.att_bias:
                self.bias += self.eta * error