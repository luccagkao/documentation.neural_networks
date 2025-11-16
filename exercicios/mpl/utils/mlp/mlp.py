import numpy as np
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

@dataclass
class TrainingHistory:
    epoch: list
    train_loss: list
    train_accuracy: list
    val_loss: list
    val_accuracy: list
    grad_norm: list
    weight_norm: list

    def as_dict(self) -> Dict[str, list]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "grad_norm": self.grad_norm,
            "weight_norm": self.weight_norm,
        }

class MLPClassifierScratch:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        hidden_layer_sizes: Iterable[int] = (16,),
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        loss: Optional[str] = None,
        eta: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 200,
        validation_fraction: float = 0.0,
        patience: int = 20,
        min_delta: float = 1e-4,
        seed: int = 42,
        weight_init: str = "auto",
        bias_as_weight: bool = False,
    ):
        self.X = self._as_2d_array(x)
        self.y_input = y
        self.n_samples, self.n_features = self.X.shape

        self.Y, self.n_classes, self.classes_, self.target_type = self._normalize_targets(y)

        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.activation = activation.lower()
        assert self.activation in {"relu","tanh","sigmoid","identity"}

        if output_activation is None:
            output_activation = "sigmoid" if self.target_type == "binary" else ("softmax" if self.target_type == "multiclass" else "tanh")
        if loss is None:
            loss = "bce" if output_activation == "sigmoid" else ("cce" if output_activation == "softmax" else "mse")

        self.output_activation = output_activation.lower()
        assert self.output_activation in {"sigmoid","softmax","tanh","identity"}
        
        self.loss_name = loss.lower()
        assert self.loss_name in {"bce","cce","mse"}

        self.eta = float(eta)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.validation_fraction = float(validation_fraction)
        self.es_patience = int(patience)
        self.es_min_delta = float(min_delta)
        self.seed = int(seed)
        
        self.weight_init = weight_init.lower()
        assert self.weight_init in {"auto","xavier","he"}
        
        self.bias_as_weight = bool(bias_as_weight)

        self.rng = np.random.default_rng(self.seed)

        if self.validation_fraction > 0.0:
            n_val = max(1, int(self.n_samples * self.validation_fraction))
            idx = self.rng.permutation(self.n_samples)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            self.Xtr, self.Ytr = self.X[tr_idx], self.Y[tr_idx]
            self.Xval, self.Yval = self.X[val_idx], self.Y[val_idx]
        else:
            self.Xtr, self.Ytr = self.X, self.Y
            self.Xval, self.Yval = None, None

        out_dim = self.n_classes if self.output_activation == "softmax" else 1

        if self.loss_name == "mse" and self.output_activation in {"tanh","identity"} and self.n_classes > 1:
            out_dim = self.n_classes

        layer_sizes = [self.n_features] + list(self.hidden_layer_sizes) + [out_dim]
        self.params = self._init_params(layer_sizes)

        self.loss_per_epoch: List[float] = []
        self.accuracy_per_epoch: List[float] = []
        self.val_loss_per_epoch: List[float] = []
        self.val_accuracy_per_epoch: List[float] = []
        self.history = TrainingHistory(epoch=[], train_loss=[], train_accuracy=[], val_loss=[], val_accuracy=[], grad_norm=[], weight_norm=[])
        self.converged: bool = False
        self.stop_reason: str = "max_epochs"
        self.epochs_run: int = 0

    @staticmethod
    def _as_2d_array(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1: x = x.reshape(-1, 1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _normalize_targets(y: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, str]:
        y_arr = np.asarray(y)
        
        if y_arr.ndim == 2 and y_arr.shape[1] > 1:
            n_classes = y_arr.shape[1]
            return y_arr.astype(np.float64), n_classes, np.arange(n_classes), "multiclass"
        
        y_flat = y_arr.reshape(-1)
        vals = np.unique(y_flat)
        
        if set(vals).issubset({0,1}) and len(vals) <= 2:
            return y_flat.astype(np.float64).reshape(-1,1), 2, np.array([0,1]), "binary"
        
        if np.issubdtype(y_flat.dtype, np.integer) and vals.min() >= 0 and (vals == np.arange(vals.max()+1)).all():
            K = int(vals.max()+1)
            onehot = np.zeros((y_flat.shape[0], K), dtype=np.float64)
            onehot[np.arange(y_flat.shape[0]), y_flat.astype(int)] = 1.0
            return onehot, K, np.arange(K), "multiclass"
        
        return y_flat.astype(np.float64).reshape(-1,1), 1, np.array([0]), "continuous"

    def _init_params(self, layer_sizes: List[int]) -> Dict[str, np.ndarray]:
        params: Dict[str, np.ndarray] = {}
        for layer_idx in range(1, len(layer_sizes)):
            fan_in = layer_sizes[layer_idx-1]
            fan_out = layer_sizes[layer_idx]
            fan_in_eff = fan_in + 1 if self.bias_as_weight else fan_in

            if self.weight_init == "auto":
                scale = np.sqrt(2.0/fan_in_eff) if self.activation == "relu" else np.sqrt(1.0/fan_in_eff)
            elif self.weight_init == "xavier":
                scale = np.sqrt(1.0/fan_in_eff)
            else:
                scale = np.sqrt(2.0/fan_in_eff)

            W = self.rng.normal(0.0, scale, size=(fan_out, fan_in_eff))
            params[f"W{layer_idx}"] = W

            if not self.bias_as_weight:
                params[f"b{layer_idx}"] = np.zeros((1, fan_out))

        return params

    # -------------------- ativações --------------------
    @staticmethod
    def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def _act(self, z):
        if self.activation == "relu":
            return np.maximum(0, z)
        
        if self.activation == "tanh":
            return np.tanh(z)
        
        if self.activation == "sigmoid":
            return self._sigmoid(z)
        
        return z

    def _act_deriv(self, a, z):
        if self.activation == "relu":
            return (z > 0).astype(z.dtype)
        
        if self.activation == "tanh":
            return 1.0 - a**2
        
        if self.activation == "sigmoid":
            return a * (1.0 - a)
        
        return np.ones_like(z)

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        preactivations: List[np.ndarray] = []
        total_layers = len(self.hidden_layer_sizes) + 1

        for layer_idx in range(1, total_layers):
            A_prev = activations[-1]
            
            if self.bias_as_weight:
                A_prev_aug = np.concatenate([A_prev, np.ones((A_prev.shape[0], 1))], axis=1)
                z_l = A_prev_aug @ self.params[f"W{layer_idx}"].T
            else:
                z_l = A_prev @ self.params[f"W{layer_idx}"].T + self.params[f"b{layer_idx}"]
            
            a_l = self._act(z_l)
            preactivations.append(z_l)
            activations.append(a_l)

        A_prev = activations[-1]

        if self.bias_as_weight:
            A_prev_aug = np.concatenate([A_prev, np.ones((A_prev.shape[0], 1))], axis=1)
            z_out = A_prev_aug @ self.params[f"W{total_layers}"].T
        else:
            z_out = A_prev @ self.params[f"W{total_layers}"].T + self.params[f"b{total_layers}"]

        if self.output_activation == "sigmoid":
            a_out = self._sigmoid(z_out)

        elif self.output_activation == "softmax":
            a_out = self._softmax(z_out)

        elif self.output_activation == "tanh":
            a_out = np.tanh(z_out)

        else:
            a_out = z_out

        preactivations.append(z_out)
        activations.append(a_out)
        
        return preactivations, activations

    def _loss_and_acc(self, Y_true: np.ndarray, Y_hat: np.ndarray) -> Tuple[float, float]:
        m = Y_true.shape[0]
        eps = 1e-12

        if self.loss_name == "bce":
            p = np.clip(Y_hat.reshape(-1,1), eps, 1-eps)
            loss = -(Y_true*np.log(p) + (1-Y_true)*np.log(1-p)).mean()

        elif self.loss_name == "cce":
            p = np.clip(Y_hat, eps, 1-eps)
            loss = -np.sum(Y_true*np.log(p)) / m

        else:
            loss = np.mean((Y_true - Y_hat)**2)

        if self.output_activation == "softmax":
            y_pred = np.argmax(Y_hat, axis=1)
            y_true = np.argmax(Y_true, axis=1)
            acc = (y_pred == y_true).mean()

        elif self.output_activation == "sigmoid":
            y_pred = (Y_hat.reshape(-1) >= 0.5).astype(int)
            y_true = Y_true.reshape(-1).astype(int)
            acc = (y_pred == y_true).mean()

        else:
            acc = np.nan

        return float(loss), float(acc)

    def _backward(self, preactivations: List[np.ndarray], activations: List[np.ndarray], Y_true: np.ndarray) -> Dict[str, np.ndarray]:
        grads: Dict[str, np.ndarray] = {}
        m = Y_true.shape[0]
        L = len(self.hidden_layer_sizes) + 1

        A_out = activations[-1]
        Z_out = preactivations[-1]
        A_prev = activations[-2]

        if self.loss_name == "bce" and self.output_activation == "sigmoid":
            dZ = A_out - Y_true

        elif self.loss_name == "cce" and self.output_activation == "softmax":
            dZ = A_out - Y_true

        elif self.loss_name == "mse":
            if self.output_activation == "tanh":
                gprime = 1.0 - np.tanh(Z_out)**2

            elif self.output_activation == "sigmoid":
                s = self._sigmoid(Z_out)
                gprime = s*(1.0 - s)

            else:
                gprime = np.ones_like(Z_out)

            dZ = 2.0*(A_out - Y_true)*gprime

        else:
            raise ValueError("Combinação de loss/saída não suportada.")

        if self.bias_as_weight:
            A_prev_aug = np.concatenate([A_prev, np.ones((m,1))], axis=1)
            grad_W = (dZ.T @ A_prev_aug) / m
            grads[f"W{L}"] = grad_W
            dA_prev = dZ @ self.params[f"W{L}"][:, :-1]
        
        else:
            grad_W = (dZ.T @ A_prev) / m
            grad_b = dZ.mean(axis=0, keepdims=True)
            grads[f"W{L}"] = grad_W
            grads[f"b{L}"] = grad_b
            dA_prev = dZ @ self.params[f"W{L}"]

        for layer_idx in range(L-1, 0, -1):
            Z_l = preactivations[layer_idx-1]
            A_l = activations[layer_idx]
            A_prev = activations[layer_idx-1]
            dG = self._act_deriv(A_l, Z_l)
            dZ = dA_prev * dG

            if self.bias_as_weight:
                A_prev_aug = np.concatenate([A_prev, np.ones((m,1))], axis=1)
                grad_W = (dZ.T @ A_prev_aug) / m
                grads[f"W{layer_idx}"] = grad_W
                if layer_idx > 1:
                    dA_prev = dZ @ self.params[f"W{layer_idx}"][:, :-1]
            
            else:
                grad_W = (dZ.T @ A_prev) / m
                grad_b = dZ.mean(axis=0, keepdims=True)
                grads[f"W{layer_idx}"] = grad_W
                grads[f"b{layer_idx}"] = grad_b
                if layer_idx > 1:
                    dA_prev = dZ @ self.params[f"W{layer_idx}"]

        return grads

    def _update(self, grads: Dict[str, np.ndarray]):
        total_layers = len(self.hidden_layer_sizes) + 1
        for layer_idx in range(1, total_layers+1):
            self.params[f"W{layer_idx}"] -= self.eta * grads[f"W{layer_idx}"]
            if not self.bias_as_weight:
                self.params[f"b{layer_idx}"] -= self.eta * grads[f"b{layer_idx}"]

    def train(self, patience: Optional[int] = None, min_delta: Optional[float] = None, verbose: bool = False):
        if patience is None: patience = self.es_patience
        if min_delta is None: min_delta = self.es_min_delta

        best = np.inf
        best_params = {k: v.copy() for k, v in self.params.items()}
        patience_count = 0

        self.loss_per_epoch.clear()
        self.accuracy_per_epoch.clear()
        self.val_loss_per_epoch.clear()
        self.val_accuracy_per_epoch.clear()
        
        self.history = TrainingHistory(epoch=[], train_loss=[], train_accuracy=[], val_loss=[], val_accuracy=[], grad_norm=[], weight_norm=[])
        
        self.stop_reason = "max_epochs"; self.converged = False

        Xtr, Ytr = self.Xtr, self.Ytr
        for epoch in range(self.max_epochs):
            idx = self.rng.permutation(len(Xtr))
            Xs, Ys = Xtr[idx], Ytr[idx]

            grad_norm_total = 0.0
            for start in range(0, len(Xs), self.batch_size):
                xb = Xs[start:start+self.batch_size]
                yb = Ys[start:start+self.batch_size]
                preacts, acts = self._forward(xb)
                grads = self._backward(preacts, acts, yb)

                for g in grads.values():
                    grad_norm_total += float(np.linalg.norm(g))
                self._update(grads)

            _, act_tr = self._forward(Xtr)
            loss_tr, acc_tr = self._loss_and_acc(Ytr, act_tr[-1])
            self.loss_per_epoch.append(loss_tr)
            self.accuracy_per_epoch.append(acc_tr if not np.isnan(acc_tr) else None)

            if self.Xval is not None:
                _, act_val = self._forward(self.Xval)
                loss_val, acc_val = self._loss_and_acc(self.Yval, act_val[-1])
                self.val_loss_per_epoch.append(loss_val)
                self.val_accuracy_per_epoch.append(acc_val if not np.isnan(acc_val) else None)
                monitor = loss_val
            else:
                loss_val = None
                acc_val = None
                monitor = loss_tr

            weight_norm_total = 0.0
            total_layers = len(self.hidden_layer_sizes) + 1

            for l in range(1, total_layers+1):
                W = self.params[f"W{l}"]
                W_use = W[:, :-1] if self.bias_as_weight else W
                weight_norm_total += float(np.linalg.norm(W_use))

            self.history.epoch.append(epoch+1)
            self.history.train_loss.append(loss_tr)
            self.history.train_accuracy.append(acc_tr if not np.isnan(acc_tr) else None)
            self.history.val_loss.append(loss_val)
            self.history.val_accuracy.append(acc_val if not np.isnan(acc_val) else None)
            self.history.grad_norm.append(grad_norm_total)
            self.history.weight_norm.append(weight_norm_total)

            if monitor + self.es_min_delta < best:
                best = float(monitor)
                best_params = {k: v.copy() for k, v in self.params.items()}
                patience_count = 0
            else:
                patience_count += 1

            if verbose:
                msg = f"Epoch {epoch+1:4d} | loss={loss_tr:.4f}"
                if acc_tr is not None: msg += f" acc={acc_tr:.3f}"
                if loss_val is not None:
                    msg += f" | val_loss={loss_val:.4f}"
                    if acc_val is not None: msg += f" val_acc={acc_val:.3f}"
                print(msg)

            if patience_count >= self.es_patience:
                self.stop_reason = f"No improvement for {self.es_patience} epochs"
                break

            self.epochs_run = epoch + 1

        self.params = best_params
        if patience_count < self.es_patience and self.epochs_run < self.max_epochs:
            self.converged = True
            self.stop_reason = "Early stopping"
        elif self.epochs_run == self.max_epochs:
            self.stop_reason = "max_epochs"

    def get_history(self) -> TrainingHistory:
        return self.history

    def history_as_dict(self) -> Dict[str, list]:
        return self.history.as_dict()

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        X = self._as_2d_array(X_test)
        _, activations = self._forward(X)
        out = activations[-1]
        if self.output_activation == "sigmoid":
            return out.reshape(-1)
        return out

    def predict(self, X_test: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X_test)
        if self.output_activation == "softmax":
            return np.argmax(proba, axis=1)
        if self.output_activation == "sigmoid":
            return (proba >= threshold).astype(int)
        return proba.reshape(-1)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        y_true = np.asarray(y_test).reshape(-1)
        y_pred = self.predict(X_test)
        
        if y_pred.ndim == 1 and y_true.ndim == 1:
            return float((y_pred == y_true).mean())
        
        return float('nan')