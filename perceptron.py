import numpy as np
from numpy.lib.function_base import select

class Perceptron:

    def __init__(self,learning_rate=0.01,number_of_iterations=1000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.activation_function = self.unit_step_function
        self.weights = None
        self.bias = None

    # X Training samples
    # y_train Traning Lables
    def fit(self,x_train,y_train):
        n_samples,n_features = x_train.shape

        #init weights
        #For each sample we add a number of features
        self.weights = np.zeros(n_features)
        self.bias = 0
        y_ = np.array([1 if i >0 else 0 for i in y_train])

        #For the number of iterations defined
        for _ in range(self.number_of_iterations):
            for index,x_i in enumerate(x_train):
                linear_output = np.dot(x_i,self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                update = self.learning_rate = (y_[index] - y_predicted)
                self.weights += update * x_i
                self.bias += update



    # X - Test Samples
    def predict(self,X):
        # The dot product
        linear_output = np.dot(X,self.weights) + self.bias
        y_predicted  = self.activation_function(linear_output)
        return y_predicted


    def unit_step_function(self,x):
        return np.where(x >= 0,1,0)
