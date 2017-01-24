import numpy as np
from sklearn.datasets import fetch_mldata
from scipy import ndimage, optimize
import matplotlib
import time

mnist = fetch_mldata('MNIST original')
X, y = mnist.data/255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


class Model():
    def __init__(self):
        #hyperparameters
        self.inputLayerSize = 28*28
        self.outputLayerSize = 10
        self.hiddenLayerSize = 28*28

        #weights
        self.W1 = np.random.randn(5, 5)
        self.W2 = np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize)

    def forward(self, X):
        #todo: validate
        yHat = np.empty([1,10])
        for x in X:
            #propagate input through network
            self.z2 = ndimage.convolve(x.reshape(28,28), self.W1, mode='constant', cval=0.0)
            self.a2 = self.sigmoid(self.z2)
            self.z3 = np.dot(self.a2.reshape(1, 784), self.W2)
            _y = self.sigmoid(self.z3)
            out = self.softmax(_y)

            yHat = np.concatenate((yHat, out), axis=0)

        return yHat

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        #derivative of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def softmax(self, y):
        self.exp_scores = np.exp(y)

        self.probs = self.exp_scores / np.sum(self.exp_scores, keepdims=True)
        return self.probs

    def calculate_loss(self, X_loss, y_loss):
        #todo: why do costFunction and calculate_loss yield different results?
        count = 0
        loss = 0
        for x in X_loss:
            self.z2 = ndimage.convolve(x.reshape(28, 28), self.W1, mode='constant', cval=0.0)
            self.a2 = self.sigmoid(self.z2)
            self.z3 = np.dot(self.a2.reshape(1, 784), self.W2)
            _y = self.sigmoid(self.z3)
            probs = self.softmax(_y)[0]
            loss -= np.log(probs[int(y_loss[count])])
            count += 1

        return (1. / len(X_loss)) * loss

    def getParams(self):
        # Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
            # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = 25
        self.W1 = np.reshape(params[W1_start:W1_end], (5, 5))
        W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def costFunction(self, X, y):
        #todo: validate
        #Compute cost for given X,y, use weights already stored in class.
        y = y.astype(int)
        self.yHat = self.forward(X)
        corect_logprobs = -np.log(self.yHat[range(len(X)), y])
        data_loss = np.sum(corect_logprobs)
        J = data_loss * (1. / len(X))
        return J

    def costFunctionPrime(self, X, y):
        #todo: implement
        dJdW1 = None
        dJdW2 = None
        return dJdW1, dJdW2

    def computeGradients(self, X, y):
        #todo: understand
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class Trainer:
    #todo: understand
    def __init__(self, N):
        self.N = N

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X, y)
        return cost, grad

    def callBackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def train(self, X, y):
        #internal variable for callback function
        self.X = X
        self.y = y

        #store costs
        self.J = []
        params0 = self.N.getParams()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0,
                                 jac=True, method='BFGS', args=(X, y),
                                 options=options, callback=self.callBackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res




model = Model()
#note: costFunction and calculate_loss yield different results even though implementation should be equal
#todo: -> SOLVE THAT!!!

start_time = time.time()
print("(1)#############################\n.costFunction method yields a weighted loss of \n%r\ncalculation took \n%r seconds" % (model.costFunction(X_test, y_test), time.time() - start_time))
start_time = time.time()
print("(2)#############################\n.calculate_loss method yields a weighted loss of \n%r\ncalculation took \n%r seconds" %(model.calculate_loss(X_test, y_test), time.time() - start_time))


# PYTHON OUT:
#
# C:\Users\Julius\Anaconda3\pythona.exe "C:/Users/Julius/PycharmProjects/simple cnn/model.py"
# (1)#############################
# .costFunction method yields a weighted loss of
# 2.4663995654964355
# calculation took
# 1.7150936126708984 seconds
# (2)#############################
# .calculate_loss method yields a weighted loss of
# 2.3962169016230606
# calculation took
# 1.3020744323730469 seconds
#
# Process finished with exit code 0







#saved for later:
# T = Trainer(model)
# T.train(X_test, y_test)
# matplotlib.plot(T.J)
# matplotlib.grid(1)
# matplotlib.xlabel('Iterations')
# matplotlib.ylabel('Cost')