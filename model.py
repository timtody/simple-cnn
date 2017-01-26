import numpy as np
from sklearn.datasets import fetch_mldata
from scipy import ndimage, optimize
import matplotlib
import time, PIL

#fetch training data
mnist = fetch_mldata('MNIST original')
#normalize training data
X, y = mnist.data/255., mnist.target
#train test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

def setupKernel(dim0, dim1, w1):
    #sets up convolution matrix with dim0xdim1 filter kernel
    #
    print("setting up convolution matrix...")
    for i in range(w1.shape[1]):
        target = [[x + i, x + i + dim1 * 2] for x in range(dim0)]
        target = [item for sublist in target for item in sublist]

        for j in range(w1.shape[0]):
            if j not in target:
                w1.itemset(j * w1.shape[1] + i, 0)
        w1 = np.around(w1, decimals=2)
    print("done.")
    return w1


class Model():
    def __init__(self):
        #hyperparameters
        self.inputLayerSize = 28*28
        self.outputLayerSize = 10
        self.hiddenLayerSize = 24*24
        self.reg_lambda = 0.01  # regularization strength
        self.epsilon = 0.01     # learning rate

        #weights
        self.w1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W1 = setupKernel(5, 5, self.w1)

        self.W2 = np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize)

    def forward(self, X):
        #todo: validate
        #propagate input through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        _y = self.sigmoid(self.z3)
        out = self.softmax(_y)

        return out

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        #derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def softmax(self, y):
        self.exp_scores = np.exp(y)

        self.probs = self.exp_scores / np.sum(self.exp_scores, keepdims=True)
        return self.probs

    def costFunction(self, X, y):
        #todo: validate
        #Compute cost for given X,y, use weights already stored in class.
        y = y.astype(int)
        self.yHat = self.forward(X)
        corect_logprobs = -np.log(self.yHat[range(len(X)), y])
        data_loss = np.sum(corect_logprobs)
        J = data_loss * (1. / (len(X)))
        return J

    def costFunctionPrime(self, X, y):
        #todo: solve issues with X.T = [0. , .... , 0.]
        y = y.astype(int)
        self.yHat = self.forward(X)
        self.yHat[range(len(X)), y] -= 1
        dW2 = (self.a2.T).dot(self.yHat)
        delta2 = self.yHat.dot(self.W2.T) * (1 - np.power(self.a2, 2))
        dW1 = np.dot(X.T, delta2)

        return dW1, dW2

    def fit(self, X, y, num):
        for i in range(num):

            #backpropagation
            dW1, dW2 = self.costFunctionPrime(X, y)


            dW1 += self.reg_lambda
            dW2 += self.reg_lambda

            # Gradient descent parameter update
            self.W1 += -self.epsilon * dW1
            self.W2 += -self.epsilon * dW2

            if i % 300 == 0:
                pass
                #print("loss at iteration %r is %r, dw1 is %r, dw2 is %r" % (i, self.costFunction(X, y), dW1, dW2))








model = Model()
model.fit(X_test, y_test, 5000)






#start_time = time.time()
#print("(1)#############################\n.costFunction method yields a weighted loss of \n%r\ncalculation took \n%r seconds" % (model.costFunction(X_test, y_test), time.time() - start_time))


# PYTHON OUT:
#
# C:\Users\Julius\Anaconda3\pythona.exe "C:/Users/Julius/PycharmProjects/simple cnn/model.py"
# (1)#############################
# .costFunction method yields a weighted loss of
# 2.4663995654964355
# calculation took
# 1.7150936126708984 seconds
#
# Process finished with exit code 0




