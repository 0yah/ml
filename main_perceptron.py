import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from perceptron import Perceptron
import pandas as pd

dataframe = pd.read_csv('pima_diabetes_dataset.txt',delimiter=" ")

"""

dataframe.values[Row,Column]
Output = dataframe.values[:5,-1]

"""


#Selects the values in the Diastolic blood pressure column
diastolic_blood_pressure_train = dataframe.values[:5,2]
diastolic_blood_pressure_test = dataframe.values[6:8,2]


#Selects the values in the Body mass index column
body_mass_index_train = dataframe.values[:5,5]
body_mass_index_test = dataframe.values[6:8,5]


"""
Returns a mapped array containing both coloumns

i.e

output =[

    [diastolic_blood_pressure[0],body_mass_index[0]],...
]

"""

x_train = np.stack((diastolic_blood_pressure_train, body_mass_index_train), axis=1)
x_test = np.stack((body_mass_index_test, body_mass_index_test), axis=1)


#print(x_train)
#Gets the last column which is the output
y_train = dataframe.values[:5,-1]
y_test = dataframe.values[:5,-1]

#print(y_train,y_test)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

p = Perceptron(learning_rate=0.01, number_of_iterations=1000)

# Feed the training data
p.fit(x_train, y_train)
predictions = p.predict(x_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=y_train)

x0_1 = np.amin(x_train[:, 0])
x0_2 = np.amax(x_train[:, 0])

x1_1 = (-p.weights[0] * x0_1 - p.bias / p.weights[1])
x1_2 = (-p.weights[0] * x0_2 - p.bias / p.weights[1])

ax.plot()

ymin = np.amin(x_train[:, 1])
ymax = np.amax(x_train[:, 1])
ax.set_ylim([ymin-3, ymax+3])

plt.show()


