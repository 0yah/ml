import numpy as np
from numpy.lib.function_base import select

class Perceptron:

    def __init__(self,learning_rate=0.01,n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_function = self._unit_step_func
        self.weights = None
        self.bias = None

    # X Training samples
    # Y Traning Lables
    def fit(self,X,y):
        n_samples,n_features = X.shape

        #init weights
        #For each sample we add a number of features
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i >0 else 0 for i in y])

        #For the number of iterations defined
        for _ in range(self.n_iters):
            for index,x_i in enumerate(X):
                linear_output = np.dot(x_i,self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                update = self.lr = (y_[index] - y_predicted)
                self.weights += update * x_i
                self.bias += update



    # X - Test Samples
    def predict(self,X):
        # The dot product
        linear_output = np.dot(X,self.weights) + self.bias
        y_predicted  = self.activation_function(linear_output)
        return y_predicted


    def _unit_step_func(self,x):
        return np.where(x >= 0,1,0)
