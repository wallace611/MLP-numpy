import numpy as np
from activations import m

class Model:
    def __init__(self):
        self._parameters: dict = {}
        self._activations: list = []
        self._cache: dict = {}
        self._grads: dict = {}
        self._depth: int = 0
        
    def init_params(self, layer_dim: list, activations: list):
        self._depth = len(layer_dim)
        self._parameters = {}
        for i in range(1, self._depth):
            fan_in = layer_dim[i - 1]
            fan_out = layer_dim[i]
            xavier_scale = np.sqrt(1.0 / fan_in)
            self._parameters[f"W{i}"] = np.random.randn(fan_out, fan_in) * xavier_scale
            self._parameters[f"b{i}"] = np.zeros((fan_out, 1))

        self._activations = [None] + activations.copy()

    def update_params(self, learning_rate: float):
        for i in range(1, self._depth):
            self._parameters[f"W{i}"] -= self._grads[f"dW{i}"] * learning_rate
            self._parameters[f"b{i}"] -= self._grads[f"db{i}"] * learning_rate
            
    def forward(self, X: np.ndarray) -> np.ndarray:
        if len(X.shape) == 1:
            tmp = X.reshape(-1, 1)
        else:
            tmp = X
        self._cache = {}
        self._cache["A0"] = tmp.copy()
        for i in range(1, self._depth):
            tmp = self._parameters[f"W{i}"] @ tmp + self._parameters[f"b{i}"]
            self._cache[f"Z{i}"] = tmp.copy()
            tmp = self._activations[i](tmp)
            self._cache[f"A{i}"] = tmp.copy()
        return tmp
    
    def backward(self, Y: np.ndarray):
        self._grads = {}
        if len(Y.shape) == 1:
            batch_size = 1
            Y = Y.reshape(-1, 1)
        else:
            batch_size = Y.shape[1]
        L = self._depth - 1
        AL = self._cache[f"A{L}"]
        delta = (AL - Y)
        self._grads[f"dW{L}"] = (delta @ self._cache[f"A{L - 1}"].T) / batch_size
        self._grads[f"db{L}"] = np.sum(delta, axis=1, keepdims=True) / batch_size
        for i in range(L - 1, 0, -1):
            delta = (self._parameters[f"W{i + 1}"].T @ delta) * m[self._activations[i]](self._cache[f"Z{i}"])
            self._grads[f"dW{i}"] = delta @ self._cache[f"A{i - 1}"].T / batch_size
            self._grads[f"db{i}"] = np.sum(delta, axis=1, keepdims=True) / batch_size

    def loss(self, AL: np.ndarray, Y: np.ndarray) -> float:
        return -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[1]

    def train(self, X_train: np.ndarray, Y_train: np.ndarray, epochs: int, batch_size: int, learning_rate: float):
        m = X_train.shape[1]
        
        losses = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            num_batches = 0
            
            for i in range(0, m, batch_size):
                end_idx = min(i + batch_size, m)
                X_batch = X_train[ : , i : end_idx]
                Y_batch = Y_train[ : , i : end_idx]
                AL = self.forward(X=X_batch)
                
                batch_loss = self.loss(AL, Y_batch)
                epoch_loss += batch_loss
                
                predict = np.argmax(AL, axis=0)
                label = np.argmax(Y_batch, axis=0)
                batch_acc = np.mean(predict == label)
                epoch_acc += batch_acc
                
                self.backward(Y_batch)
                
                self.update_params(learning_rate)
                
                num_batches += 1
            
            epoch_loss /= num_batches
            epoch_acc /= num_batches
            
            print(f"Epoch {epoch + 1} / {epochs}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
            losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            
        return losses, accuracies
    
        