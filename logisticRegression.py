import numpy as np
import csv
from math import exp
from math import log
import math

# Class 4 Slide 15
def logistic ( WT, x ):

	WTx = np.asscalar(np.dot(WT, x))

	# These following checks are made, or else if sigmoid returns 0 or 1, the error function will fail
	try:
		temp = math.e**(-WTx)
	except Exception as e:
		return 2.2250738585072014e-308 # exp(-WTx) is very big, so the sigmoid function would return something very small.

	if (float(1) / (float(1) + exp(-WTx)) == 1.0): # exp(-WTx) is very small, so the sigmoid function would return something close to 1
		return 0.9999999999999999

	return float(1) / (float(1) + exp(-WTx))


# Class 4 Slide 17
def error ( X, Y, W ):

	WT = np.transpose(W)

	sum = 0.0
	for x, y in zip(X, Y):
		x = np.transpose(x)

		logFunV = logistic(WT, x)
		try:
			sum += np.asscalar(y) * log(logFunV)
			sum += (float(1) - np.asscalar(y)) * log(float(1) - logFunV)
		except Exception as e:
			raise e

	return -sum


# Class 4 Slide 17
def getNewW ( X, Y, W, alpha ):
	
	WT = np.transpose(W)

	sum = 0.0
	for x, y in zip(X, Y):
		x = np.transpose(x)

		logFunV = logistic(WT, x)

		sum += np.dot(x, np.asscalar(y) - logFunV)

	return W + alpha * sum


def logisticRegression( X, Y, alpha ):

	# Normalize Data
	meanX = (np.mean(X, axis=0))
	stdX = np.std(X, axis=0)

	X = (X - meanX) / stdX

	# Insert a column of x0 = 1
	(row, col) = X.shape
	X = np.insert(X, col, 1.0, axis=1)

	# Initialize W as matrix of 0s
	shape = X.shape[1]	

	W = np.zeros(shape)
	W = np.matrix(W)
	W = np.transpose(W)

	# Find initial error with W = [0]
	errorV = error(X, Y, W)

	# Dummy value so that 10 > 0.5
	deltaError = 10

	# Iteration counter
	i = 0

	# If deltaError ever becomes negative AND/OR becomes smaller than the threshold, stop
	while(deltaError > 0.5):

		print(i)
		i += 1

		# Save old error
		oldErrorV = errorV

		# Get new set of W
		W = getNewW(X, Y, W, alpha)

		# Get new error with new set of W
		errorV = error(X, Y, W)

		# Find error difference
		deltaError = oldErrorV - errorV

		print("olderrorV " + str(oldErrorV))
		print("errorV " + str(errorV))
		print("deltaError " + str(deltaError))

	return W
	

def main():
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

	data = data[1:]	
	result = result[1:]

	# remove ID from all entries
	# convert to float
	parsedData = []
	parsedResult = []

	for entry in data:
		entry.pop(0)
		entry = [float(i) for i in entry]
		parsedData.append(entry)

	for entry in result:
		entry.pop(0)
		parsedResult.append(float(entry[0]))

	# X data
	X = np.matrix(parsedData)

	# Y result
	Y = np.matrix(parsedResult).transpose()

	# alpha: step for gradientDescent
	alpha = 0.0001

	# Weights
	W = logisticRegression(X, Y, alpha)

	print(W)

main()
