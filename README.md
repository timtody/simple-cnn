
# A rudimentary Convolutional Neural Network in Python
This post consists of my implementation of a three layer Convolutional Neural Network in Python. Please note that this network is neither effiecient nor achieves state of the art classification performance. Since the purpose is solely educational i tried to keep it as basic as possible with as little as possible dependencies.

## Getting Started
### Dependencies
#### I strongly advise using the Anaconda3 distribution which contains all of the following dependencies

* [Anaconda](https://www.continuum.io/downloads) - The Anaconda Python distribution

#### Alternatively you can install the dependencies by hand
* [Python](https://www.python.org/) - The programming environment
* [Numpy](http://www.scipy.org/scipylib/download.html)
* [Scipy](http://www.scipy.org/scipylib/download.html)
* [scikit-learn](http://scikit-learn.org/stable/install.html)
* [matplotlib](http://matplotlib.org/users/installing.html)

### Training a model
First instantiate a model object and pass reg_lambda (regularization strength) and epsilon (learning rate) which default to **0.009** and **0.00001** respectively.
```
model = Model(reg_lamda=0.01, epsilon=0.00005)
```
Then call the fit method passing the training data and the number of epochs (20-30 seems like a good choice, depending on your machine).
```
model.fit(X_train, y_train, 30)
```
### Testing your model
You can check the performance of your model after training:
```
model.test(X_test, y_test)
```
which uses the now trained parameters to predict labels for the test data X_test and compares them to the correct labels in y_test.
### Save weights for later use
You can save the trained parameters as numpy arrays to /nets with
```
model.dumpParams()
```
and load previously saved nets with
```
model.fromParams(nets/nameOfYourNet)
```
which overrides the current weight matrices W1 and W2 of model.
