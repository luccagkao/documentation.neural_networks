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

    def train(self, patience: int = 10, min_delta: int = 0):
        best_updates = np.inf
        best_weights = self.weights.copy()
        best_bias = float(self.bias)
        patience_count = 0

        self.stop_reason = "max_epochs"
        self.converged = False
        self.accuracy_per_epoch = []
        self.updates_per_epoch = []

        for epoch in range(self.max_epochs):
            updates = 0

            indices = np.random.permutation(len(self.x))
            x_shuffled = self.x[indices]
            y_shuffled = self.y[indices]

            for x_i, y_i in zip(x_shuffled, y_shuffled):
                y_pred = 1 if (np.dot(x_i, self.weights) + self.bias) >= 0 else 0
                error = y_i - y_pred
                if error != 0:
                    updates += 1
                    self._rebalance(x=x_i, error=error)

            acc = 1.0 - (updates / len(self.y))
            self.accuracy_per_epoch.append(acc)
            self.updates_per_epoch.append(updates)
            self.epochs_run = epoch + 1

            if updates == 0:
                self.converged = True
                self.stop_reason = "No errors (linearly separable)"

                best_updates = 0
                best_weights = self.weights.copy()
                best_bias = float(self.bias)
                break

            if updates + min_delta < best_updates:
                best_updates = updates
                best_weights = self.weights.copy()
                best_bias = float(self.bias)
                patience_count = 0  # reset paciência
            else:
                patience_count += 1

            if patience_count >= patience:
                self.stop_reason = f"No improvement for {patience} epochs"
                break

        self.weights = best_weights
        self.bias = best_bias
        self.best_updates = best_updates

    
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