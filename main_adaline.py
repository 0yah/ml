from sklearn.datasets import load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from adaline import Adaline



bc = load_breast_cancer()

X = bc.data
y = bc.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

adaline = Adaline(number_of_iterations=10)



adaline.fit(X_train, y_train)

score = adaline.score(X_test, y_test)

print(score)