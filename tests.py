import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

"""
sklearn.model_selection.train_test_split(*input_arrays,**options)


input_arrays are sequences of lists,NumPy arrays, panda DataFrams or similar array-like objects
They must be of the same length

In Supervised learning we use

2D input array
1D output array

**Options

train_size - Size of the training size. A float value provided must be between 0.0 and 1.0. Int provided will represent the total number of training samples

test_size - Size of the test set. Default value is 0.25 or 25 percent

random_state - Controls randomization during splitting. Can be an int or an instance of the RandomState class

shuffle - Determines whether to shuffle the dataset before applying the split

stratify - Array like object that Determines how to stratify a split


X - Input 
y - Output
"""


x = np.arange(1,25).reshape(12,2)
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
print(x,y)

#Splitting
"""

x_train - The training inputs of the first sequence(x)
x_test - The testing inputs of the first sequence(x)

y_train - The training outputs of the first sequence(x)
y_test - The testing outputs of the first sequence(x)

"""
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=4,random_state=4)


print("\nTraining Inputs\n {} \n".format(x_train))
print("Training Outputs\n {} \n".format(y_train))

print("Testing Inputs\n {} \n".format(x_test))
print("Testing Outputs\n {} \n".format(y_test))