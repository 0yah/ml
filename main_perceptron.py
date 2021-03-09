import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from perceptron import Perceptron


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


X, y = datasets.make_blobs(n_samples=150, n_features=2,
                           centers=2, cluster_std=1.05, random_state=2)


#############################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

#############################

p = Perceptron(learning_rate=0.01, number_of_iterations=1000)

# Feed the training data
p.fit(X_train, y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-p.weights[0] * x0_1 - p.bias / p.weights[1])
x1_2 = (-p.weights[0] * x0_2 - p.bias / p.weights[1])

ax.plot()

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin-3, ymax+3])

plt.show()


class Adaline(object):

    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):

        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate




    '''
    Batch Gradient Descent
    1. Weights are updated considering all training examples.
    2. Learning of weights can continue for multiple iterations
    3. Learning rate needs to be defined
    '''
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        for _ in range(self.n_iterations):
            activation_function_output = self.activation_function(
                self.net_input(X))
            errors = y - activation_function_output
            self.coef_[1:] = self.coef_[1:] + \
                self.learning_rate*X.T.dot(errors)
            self.coef_[0] = self.coef_[0] + self.learning_rate*errors.sum()

    '''

    Net Input is sum of weighted input signals

    '''

    def net_input(self, X):
        weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
        return weighted_sum

    '''

    Activation function is fed the net input. As the activation function is

    an identity function, the output from activation function is same as the

    input to the function.

    '''

    def activation_function(self, X):

        return X

    '''

    Prediction is made on the basis of output of activation function

    '''

    def predict(self, X):
        return np.where(self.activation_function(self.net_input(X)) >= 0.0, 1, 0)

    '''

    Model score is calculated based on comparison of 

    expected value and predicted value

    '''

    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if(target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_
