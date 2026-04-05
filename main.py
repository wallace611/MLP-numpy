from model import Model
from activations import relu, softmax
from sklearn.datasets import fetch_openml
import numpy as np

print("Fetching datasets...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X = mnist.data
Y_raw = mnist.target.astype(np.int64)
Y = np.zeros((10, len(Y_raw)))
Y[Y_raw, np.arange(len(Y_raw))] = 1
X = (X / 255.0).T
print("Done.")

m = Model()
m.init_params([784, 128, 64, 10], [relu, relu, softmax])
m.train(X, Y, 100, 128, 0.05)