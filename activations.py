import numpy as np

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))
    
def sigmoid_d(A: np.ndarray) -> np.ndarray:
    return sigmoid(A) * (1 - sigmoid(A))
    
def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)
    
def relu_d(A: np.ndarray) -> np.ndarray:
    return (A > 0).astype(int)

def softmax(Z: np.ndarray) -> np.ndarray:
    max_Z = np.max(Z, axis=0, keepdims=True)
    Z_shifted = Z - max_Z
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
def softmax_d(A: np.ndarray) -> np.ndarray:
    ...
    
m = {
    sigmoid: sigmoid_d,
    relu: relu_d,
    softmax: softmax_d
}