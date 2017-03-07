
# A rudimentary Convolutional Neural Network in Python
This post consists of my implementation of a three layer Convolutional Neural Network in Python. Please note that this network is neither effiecient nor archieves state of the art classification performance. Since the purpose is solely educational i tried to keep it as basic as possible with as little as possible dependencies.

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
First instantiate a model object
```
model = Model()
```
Then call the fit method passing the training data and the number of epochs (20-30 seems a good number, depending on your machine)
```
model.fit(X_train, y_train, 30)
```

If you want you can tweak the hyperparameters in the code
```
self.reg_lambda = 0.009  # regularization strength
self.epsilon = 0.00001  # learning rate
```
