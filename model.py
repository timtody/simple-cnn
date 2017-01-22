import numpy as np
from sklearn.datasets import fetch_mldata
from scipy import ndimage

mnist = fetch_mldata('MNIST original')

class Model():
    def __init__(self):
        #hyperparameters
        self.inputLayerSize = 28*28
        self.outputLayerSize = 10
        self.hiddenLayerSize = 28*28

        #weights
        self.kernel = np.random.randn(5, 5)
        self.W2 = np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize)


    def forward(self, X):
        #propagate input through network
        self.z2 = ndimage.convolve(X, self.kernel, mode='constant', cval=0.0)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        _y = self.sigmoid(self.z3)
        return self.softmax(_y)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def softmax(self, y):
        self.exp_scores = np.exp(y)

        self.probs = self.exp_scores / np.sum(self.exp_scores, keepdims=True)
        return self.probs


model = Model()
model.forward(mnist.data[0])
