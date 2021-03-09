import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


# Adaline is a single layer neural network

# Net Input function -> Linear Activation Function -> Threshold Function

"""
Threshold Function 
It is binary function based on unit step function
The output of the activation function which is net input is compared with 0 and the outout is 1 or 0 depending on whether the net input is greater thatn or equal to 0

"""


class Adaline(object):

    def __init__(self, number_of_iterations=100, random_state=1, learning_rate=0.01):

        self.number_of_iterations = number_of_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    """
    Unlike Perceptrons whre weights are learned based on the prediction value whic is derived as the output of the unit step function, the weights of adaline are 
    learned by comparing the actual/ expected value with the out of activation function which is a continous value

    *The weights are based on batch gradient descent algorithm which requires the weights to be updated after considering the weight updates related to all training examples

    Batch Gradient Descent 

    1. Weights are updated considering all traing examples
    2. Learning of weights can contine for multiple iterations
    3. The learning rate needs to be defined when this class is being instanciated
    """

    def fit(self, X_train, Y_train):
        rgen = np.random.RandomState(self.random_state)
        self.weight = rgen.normal(loc=0.0, scale=0.01, size=1 + X_train.shape[1])
        for _ in range(self.number_of_iterations):

            activation_function_output = self.activation_function(self.net_input(X_train))

            errors = Y_train - activation_function_output

            #Update the weight
            self.weight[1:] = self.weight[1:] + \
                self.learning_rate * X_train.T.dot(errors)
            
            #Update the Bias
            self.weight[0] = self.weight[0] + self.learning_rate*errors.sum()

    """

    Net Input function is a combination of the input signals of different weights
    input_i = x_i * w_i
    net_input = sum of input_i
    The net input is fed into the activation function which is returns binary results(1 or 0) based on a condition

    """

    def net_input(self, X):
        
        """
        
        Bias - self.weight[0]
        Weight - self.weight[1:]
        
        """
        
        weighted_sum = np.dot(X, self.weight[1:]) + self.weight[0]
        return weighted_sum

    """

    The net input is fed into activation function to calculate the output.
    The activation function is a linear activation function.
    the output is the same as the input(identity function)
    The activation function output is used to learn weights
    The output of this function is used to calculate the change in wieghts related to different inputs which will be update to learn new weights

    """

    def activation_function(self, X):
        #Linear activation
        return X

    """

    Prediction is based in the unit step function which provides output as  1 or 0 based on whether the output of the activation function is greater thaan or equal to zero
    if the output is greater than or equal to zero then the prediction is 1 or else 0

    """

    def predict(self, x_test):
        """

        if output > 0:

        """
        return np.where(self.activation_function(self.net_input(x_test)) >= 0.0, 1, 0)

    '''

    Model score is calculated based on comparison of expected value and predicted value

    '''

    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if(target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count) / total_data_count
        return self.score_
