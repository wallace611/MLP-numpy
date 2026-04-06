from model import Model
from activations import relu, softmax
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

def get_datasets():
    print("Fetching datasets...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data
    Y_raw = mnist.target.astype(np.int64)
    Y = np.zeros((10, len(Y_raw)))
    Y[Y_raw, np.arange(len(Y_raw))] = 1
    X = (X / 255.0).T
    n = X.shape[1]
    print("Done.")
    return X, Y, n

def plot_training_history(losses: list, accuracies: list):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs_range = range(1, len(losses) + 1)
    
    ax1.plot(epochs_range, losses, color='red', marker='o', markersize=3, label='Training Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (Cross-Entropy)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    ax2.plot(epochs_range, accuracies, color='blue', marker='o', markersize=3, label='Training Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

X, Y, n = get_datasets()
slic = int(n * 6 / 7)
X_train, X_test = X[:, : slic], X[:, slic :]
Y_train, Y_test = Y[:, : slic], Y[:, slic :]

m = Model()
m.init_params([784, 128, 64, 10], [relu, relu, softmax])
losses, accuracies = m.train(X_train, Y_train, 200, 128, 0.05)

AL = m.forward(X_test)
print(m.accuracy(AL, Y_test))

plot_training_history(losses, accuracies)