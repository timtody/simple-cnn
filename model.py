import numpy as np
import scipy
from sklearn.datasets import fetch_openml
from tempfile import TemporaryFile
import matplotlib.pyplot as plt
import time, pickle, warnings

warnings.filterwarnings('error')
try:
    warnings.warn(Warning())
except Warning:
    print('Warning was raised as an exception!')
#np.seterr(all='raise')

#fetch training data
mnist = fetch_openml('mnist_784', version=1)
#normalize training data
X, y = mnist.data/255., mnist.target
#train test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

def setupKernel(dim0, dim1, w1):
    #sets up convolution matrix with dim0xdim1 filter kernel
    #
    print("setting up convolution matrix...")
    weights = np.random.randn(dim0*dim1) / np.sqrt(dim0)

    for i in range(w1.shape[1]):
        target = [[x + i, x + i + dim1 * 2] for x in range(dim0)]
        target = [item for sublist in target for item in sublist]
        count = 0
        for j in range(w1.shape[0]):
            if j in target:
                w1.itemset(j * w1.shape[1] + i, weights[count])
                count += 1
    print("done.")
    return w1


class Model():
    def __init__(self, reg_lambda=0.009, epsilon=0.00001):
        #hyperparameters
        self.inputLayerSize = 28*28
        self.outputLayerSize = 10
        self.hiddenLayerSize = 24*24
        self.reg_lambda = reg_lambda # regularization strength
        self.epsilon = epsilon  # learning rate

        #weights
        self.w1 = np.zeros((self.inputLayerSize, self.hiddenLayerSize)) / np.sqrt(self.inputLayerSize)
        self.W1 = setupKernel(5, 5, self.w1)**2
        self.b1 = np.zeros((1, self.hiddenLayerSize))

        self.W2 = abs(np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize) / np.sqrt(self.hiddenLayerSize))**2
        self.b2 = np.zeros((1, self.outputLayerSize))

    def forward(self, X):
        #propagate input through network
        #W1 is convolution matrix
        self.z2 = np.dot(X, self.W1)
        self.a2 = np.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        _y = self.z3
        out = self.softmax(_y)
        return out

    def sigmoid(self, z):
        #activation function
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        #derivative of sigmoid activation function
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def tanh_deriv(self, x):
        #derivative of alternative acitivation function
        return 1.0 - np.tanh(x) ** 2

    def softmax(self, X):
        #softmax normalozation
        exp_scores = np.exp(X)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def test(self, X, y):
        num = len(X)
        right = 0
        yHat = np.argmax(self.forward(X), axis=1)
        y = y.astype(int)

        for i, e in enumerate(yHat):
            if e == y[i]:
                right += 1

        return right / num

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        y = y.astype(int)
        yHat = self.forward(X)
        rng = np.random.randint(0, 60000)
        corect_logprobs = -np.log(yHat[range(len(X)), y])
        data_loss = np.sum(corect_logprobs)
        J = data_loss * (1. / (len(X)))

        return J

    def costFunctionPrime(self, X, y):
        y = y.astype(int)
        delta3 = self.forward(X)
        delta3[range(len(X)), y] -= 1
        dW2 = (self.a2.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(self.W2.T) * (1 - np.power(self.a2, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)








        # delta3 = self.forward(X)

        #delta3[range(len(X)), y] -= 1



        # delta3 = np.multiply(delta3, self.tanh_deriv(self.z3))
        #
        # for x in delta3:
        #     print(x)
        #
        # dW2 = (self.a2.T).dot(delta3)
        # db2 = np.sum(delta3, axis=0, keepdims=True)
        #
        # delta2 = np.dot(delta3, self.W2.T) * self.tanh_deriv(self.z2)
        #
        # dW1 = np.dot(X.T, delta2)
        # db1 = np.sum(delta2, axis=0)


        return dW1, dW2, db1, db2

    def fit(self, X, y, num, print_loss=False, print_frequence=1):
        errors = []
        iterations = []
        loss_previous = self.costFunction(X, y)
        for i in range(num):
            #self.epsilon = self.epsilon * 0.99
            if i % print_frequence == 0 and print_loss:
                #decrease learning rate
                self.epsilon = self.epsilon * 0.9
                loss = self.costFunction(X, y)
                print("Iteration %r: loss: %r" % (i, round(loss, ndigits=6)))
                errors.append(loss)
                iterations.append(i)
                loss_previous = loss
                
                print("Accuracy (training): %r percent" % ( round(100*self.test(X_train, y_train),ndigits=4)))
                print("Accuracy (test):  %r percent" % ( round(100*self.test(X_test, y_test),ndigits=4)))

            # backpropagation
            dW1, dW2, db1, db2 =  self.costFunctionPrime(X, y)
            
            #only dW2 can be regularized because of matrix represenation of convolutions
            dW2 += self.reg_lambda


            # gradient descent parameter update
            self.W1 += -self.epsilon * dW1
            self.b1 += -self.epsilon * db1
            self.W2 += -self.epsilon * dW2
            self.b2 += -self.epsilon * db2
        
        #plot results
        plt.plot(iterations, errors)
        plt.yscale('log')
        plt.ylabel("Error (log) ")
        plt.xlabel("Iterations")
        plt.show()

    def dumpParams(self):
        outfile = TemporaryFile(delete=False, dir='nets')
        np.savez(outfile, W1=self.W1, W2=self.W2, b1=self.b1, b2=self.b2)
        print("hi")

    def fromParams(self, infile):
        npzfile = np.load(infile)
        self.W1 = npzfile['W1']
        self.W2 = npzfile['W2']
        self.b1 = npzfile['b1']
        self.b2 = npzfile['b2']

    def predict(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = np.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        _y = self.z3
        exp_scores = np.exp(_y)
        probs = exp_scores / np.sum(exp_scores,  keepdims=True)

        return np.argmax(probs)






if __name__ == "__main__":
    model = Model(reg_lambda=0.01, epsilon=0.00005)
    model.fit(X_train, y_train, 30)
    model.test(X_test, y_test)






