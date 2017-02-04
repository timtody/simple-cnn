import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import time, pickle, warnings

warnings.filterwarnings('error')
try:
    warnings.warn(Warning())
except Warning:
    print('Warning was raised as an exception!')
#np.seterr(all='raise')

#fetch training data
mnist = fetch_mldata('MNIST original')
#normalize training data
X, y = mnist.data/255., mnist.target
#train test split
X_train, X_test = X[50000:60000], X[65000:]
y_train, y_test = y[50000:60000], y[65000:]

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
    def __init__(self):
        #hyperparameters
        self.inputLayerSize = 28*28
        self.outputLayerSize = 10
        self.hiddenLayerSize = 24*24
        self.reg_lambda = 0.9  # regularization strength
        self.epsilon = 0.0003  # learning rate

        #weights
        self.w1 = np.zeros((self.inputLayerSize, self.hiddenLayerSize)) / np.sqrt(self.inputLayerSize)
        self.W1 = setupKernel(5, 5, self.w1)**2
        self.b1 = np.zeros((1, self.hiddenLayerSize))

        self.W2 = abs(np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize) / np.sqrt(self.hiddenLayerSize))**2
        self.b2 = np.zeros((1, self.outputLayerSize))

    def forward(self, X):
        #todo: validate
        #propagate input through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = np.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        _y = self.z3
        out = self.softmax(_y)

        return out

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
                #derivative of sigmoid function
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def tanh_deriv(self, x):
        return 1.0 - np.tanh(x) ** 2

    def softmax(self, y):

        exp_scores = np.exp(y)
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
        #todo: validate
        #Compute cost for given X,y, use weights already stored in class.
        y = y.astype(int)
        self.yHat = self.forward(X)
        corect_logprobs = -np.log(self.yHat[range(len(X)), y])
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

    def fit(self, X, y, num):
        errors = []
        iterations = []
        loss_previous = self.costFunction(X, y)
        for i in range(num):
            self.epsilon = self.epsilon * 0.99



            if i % 1 == 0:
                #self.epsilon = self.epsilon * 0.7
                # print("loss at iteration %r is %r\ndw1 is %r, dw2 is %r" % (i, self.costFunction(X, y), dW1, dW2))
                loss = self.costFunction(X, y)
                print("loss at iteration %r has decreased by %r and loss is %r" % (i, (loss_previous-loss), loss))
                errors.append(loss)
                iterations.append(i)
                loss_previous = loss

                print(self.test(X_train, y_train))
                print(self.test(X_test, y_test))

            # backpropagation
            dW1, dW2, db1, db2 =  self.costFunctionPrime(X, y)
            #print(dW1, dW2)
            #dW1 += self.reg_lambda
            dW2 += self.reg_lambda


            #print(dW1.shape, dW2.shape)

            # gradient descent parameter update
            self.W1 += -self.epsilon * dW1
            self.b1 += -self.epsilon * db1
            self.W2 += -self.epsilon * dW2
            self.b2 += -self.epsilon * db2

        plt.plot(iterations, errors)
        plt.yscale('log')
        plt.ylabel("Error (log) ")
        plt.xlabel("Iterations")
        #plt.grid(1)
        plt.show()



model = Model()
#print(model.costFunction(X, y))
model.fit(X_train, y_train, 10)
res = model.test(X_test, y_test)

print(res)






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




