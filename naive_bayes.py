from math import exp, sqrt, pi
from numpy import random
import pandas as pd
import numpy as np
from csv import reader

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Separate training data by class
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()

        separated[class_value].append(vector)
    return separated
    
# Get the mean from the dataset
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# Get the standard deviation
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers])/float(len(numbers)-1)
    return sqrt(variance)

# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column))for column in zip(*dataset)]
    del(summaries[-1])
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# Calculate the gaussian probability distribution function for x
def calculate_probalility(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2*pi) * stdev)) * exponent

def calculate_class_probabilities(summaries,row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)

        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probalility(row[i],mean,stdev)
    
    return probabilities

def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

if __name__ == "__main__":

    dataframe = pd.read_csv('pima_diabetes_dataset.txt',delimiter=" ")
    dataset = dataframe.values
    #Get the first 600 rows as the train data
    train_dataset = dataset[:600]

    #Get the remain rows as the test data
    test_dataset = dataset[600:]
    
    model = summarize_by_class(train_dataset)

    #Select a random row in the test dataset
    random_row=random.randint(0,high=168)
    
    # predict the label
    label = predict(model, test_dataset[random_row])

    print('Data=%s, Predicted: %s' % (test_dataset[random_row], label))
    
    