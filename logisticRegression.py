import numpy as np
import csv
from math import exp
from math import log

# Class 4 Slide 15
def logistic ( WT, x ):

	print(np.shape(WT))
	print(np.shape(x))

	WTx = np.dot(WT, x)

	print(WTx)

	e = exp(-WTx)

	return (1.0 / (1.0 + e))


# Class 4 Slide 17
def cost ( X, Y, W ):

	WT = np.transpose(W)

	sum = 0.0
	for x, y in zip(X, Y):
		x = np.transpose(x)
		sum += y * log(logistic(WT, x))
		sum += (1.0 - y) * log(1 - logistic(WT, x))

	return -sum


# Class 4 Slide 17
def getNewW ( X, Y, W, alpha ):
	
	WT = np.transpose(W)

	sum = 0.0
	for x, y in zip(X, Y):
		x = np.transpose(x)
		sum += np.dot(x, (y - logistic(WT, x)))

	return W - alpha * sum


def logisticRegression( X, Y, alpha ):

	#W = findWeights(X, Y)

	shape = X.shape[1]

	W = np.zeros(shape)

	W = np.matrix(W)
	W = np.transpose(W)

	W = getNewW(X, Y, W, alpha)

	costV = cost(X, Y, W)

	changeCost = 1

	i = 0

	while(changeCost > 0.001):

		print(i)
		i += 1

		W = getNewW(X, Y, W, alpha)

		#print(W)

		oldCostV = costV
		costV = cost(X, Y, W)
		changeCost = oldCostV - costV


	return W
	


data = []
result = []
with open('parsedOutput.csv','rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

with open('result.csv','rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        result.append(row)

data = data[1:2]	
result = result[1:2]

# remove ID from all entries
# convert to float
parsedData = []
parsedResult = []

for entry in data:
	entry.pop(0)
	entry = [float(i) for i in entry]
	entry.append(1.0) # x0 = 1
	parsedData.append(entry)

for entry in result:
	entry.pop(0)
	parsedResult.append(float(entry[0]))

# X data
X = np.matrix(parsedData)

# Y result
Y = np.matrix(parsedResult)

# alpha: step for gradientDescent
alpha = 0.001

# Weights
W = logisticRegression(X, Y, alpha)