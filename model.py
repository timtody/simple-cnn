import numpy as np
from sklearn.datasets import fetch_mldata
from scipy import ndimage

mnist = fetch_mldata('MNIST original')
X, y = mnist.data , mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


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
        self.z2 = ndimage.convolve(X.reshape(28,28), self.kernel, mode='constant', cval=0.0)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2.reshape(1, 784), self.W2)
        _y = self.sigmoid(self.z3)
        return np.argmax(self.softmax(_y), axis=1)

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(z):
        #derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)



    def softmax(self, y):
        self.exp_scores = np.exp(y)

        self.probs = self.exp_scores / np.sum(self.exp_scores, keepdims=True)
        return self.probs

    def claculate_loss(self, X, y):
        count = 0
        loss = 0
        for x in X:
            self.z2 = ndimage.convolve(x.reshape(28, 28), self.kernel, mode='constant', cval=0.0)
            self.a2 = self.sigmoid(self.z2)
            self.z3 = np.dot(self.a2.reshape(1, 784), self.W2)
            _y = self.sigmoid(self.z3)
            probs = self.softmax(_y)
            loss += y[count] * np.log(probs)
            count += 1

        sum = -np.sum(loss)
        return (1. / len(X)) * sum






model = Model()
print(model.forward(X_test[0]))
print(model.claculate_loss(X_test,y_test))
