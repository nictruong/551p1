import numpy as np
import csv
from math import exp
from math import log
import math
import matplotlib.pyplot as plt
import random

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


def logisticRegression( X, Y, alpha, errorThres ):

	X = normalizeAndAddX0(X)

	X, Y = shuffle(X, Y)

	# Initialize W as matrix of 0s
	shape = X.shape[1]	

	W = np.zeros(shape)
	W = np.matrix(W)
	W = np.transpose(W)

	# Find initial error with W = [0]
	errorV = error(X, Y, W)

	# Dummy value so that 10 > errorThres
	deltaError = 10

	# Iteration counter
	i = 0

	iteration = []
	errorList = []

	# If deltaError ever becomes negative AND/OR becomes smaller than the threshold, stop
	while(deltaError > errorThres):

		temp = W

		# print(i)

		# Save old error
		oldErrorV = errorV

		# Get new set of W
		W = getNewW(X, Y, W, alpha)

		# Get new error with new set of W
		errorV = error(X, Y, W)

		# Find error difference
		deltaError = oldErrorV - errorV

		if (deltaError < 0):
			W = temp
			break

		# print("olderrorV " + str(oldErrorV))
		# print("errorV " + str(errorV))
		# print("deltaError " + str(deltaError))

		iteration.append(i)
		errorList.append(errorV)

		i += 1

	return (W, iteration, errorList)

def normalizeAndAddX0(X):

	# Normalize Data
	meanX = (np.mean(X, axis=0))
	stdX = np.std(X, axis=0)

	X = (X - meanX) / stdX

	# Insert a column of x0 = 1
	(row, col) = X.shape
	X = np.insert(X, col, 1.0, axis=1)

	return X

def shuffle(X, Y):

	randomize = np.arange(len(X))
	np.random.seed()
	np.random.shuffle(randomize)
	X = X[randomize]
	Y = Y[randomize]
	
	return X, Y

def kFoldTesting(X, Y, alpha, errorThres):
	# number of partitions
	k = 2

	# Shuffle the X and Y
	(X, Y) = shuffle(X, Y)

	counter = 0
	for y in Y:
		if y == 1:
			counter += 1

	print counter

	nbEntries = len(X)

	interval = nbEntries / k

	start = 0
	end = interval

	stats = []

	for i in range(k):

		start = interval * i
		end = interval * (i + 1)

		trainingX1 = X[0:start]
		trainingY1 = Y[0:start]

		trainingX2 = X[end + 1:]
		trainingY2 = Y[end + 1:]

		validationX = normalizeAndAddX0(X[start+1:end])
		validationY = Y[start+1:end]

		trainingX = np.vstack((trainingX1, trainingX2))
		trainingY = np.vstack((trainingY1, trainingY2))

		(trainingW, iter, errorList) = logisticRegression(trainingX, trainingY, alpha, errorThres)
		trainError = error(normalizeAndAddX0(trainingX), trainingY, trainingW)


		# STATS
		class1 = 0
		class0 = 0

		class1Guess = 0
		class0Guess = 0
		
		m11 = 0
		m00 = 0

		m10 = 0
		m01 = 0



		trainingWT = np.transpose(trainingW)

		print(trainingWT)

		for x, y in zip(validationX, validationY):

			x = np.transpose(x)

			if y == 1:
				class1 += 1
			elif y == 0:
				class0 += 1

			logFunV = logistic(trainingWT, x)

			if (logFunV > 0.5 and y == 1):
				m11 += 1

			if (logFunV <= 0.5 and y == 0):
				m00 += 1			

			if (logFunV <= 0.5 and y == 1):
				m10 += 1

			if (logFunV > 0.5 and y == 0):
				m01 += 1	

			if (logFunV > 0.5):
				class1Guess += 1

			if (logFunV <= 0.5):
				class0Guess += 1

		validError = error(validationX, validationY, trainingW)

		stats.append([k, trainError, validError, class1, class0, class1Guess, class0Guess, m11, m00, m10, m01])


	with open("stats.csv", "wt") as f:
		writer = csv.writer(f)
		writer.writerow(["k", "trainError", "validError", "class1", "class0", "class1Guess", "class0Guess", "m11", "m00", "m10", "m01"])
		writer.writerows(stats)


def test(X, Y, W):

	(X, Y) = shuffle(X, Y)

	X = normalizeAndAddX0(X)

	WT = np.transpose(W)

	# STATS
	class1 = 0
	class0 = 0

	class1Guess = 0
	class0Guess = 0
	
	m11 = 0
	m00 = 0

	m10 = 0
	m01 = 0

	stats = []

	for x, y in zip(X, Y):

		x = np.transpose(x)

		if y == 1:
			class1 += 1
		elif y == 0:
			class0 += 1


		logFunV = logistic(WT, x)

		if (logFunV > 0.5 and y == 1):
			m11 += 1

		if (logFunV <= 0.5 and y == 0):
			m00 += 1			

		if (logFunV <= 0.5 and y == 1):
			m10 += 1

		if (logFunV > 0.5 and y == 0):
			m01 += 1	

		if (logFunV > 0.5):
			class1Guess += 1

		if (logFunV <= 0.5):
			class0Guess += 1

	errorV = error(X, Y, W)

	stats.append([errorV, class1, class0, class1Guess, class0Guess, m11, m00, m10, m01])

	with open("teststats.csv", "wt") as f:
		writer = csv.writer(f)
		writer.writerow(["error", "class1", "class0", "class1Guess", "class0Guess", "m11", "m00", "m10", "m01"])
		writer.writerows(stats)

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

	# remove ID from all entries
	# convert to float
	parsedData = []
	parsedResult = []

	data = data[1:]
	result = result[1:]

	for entry in data:
		entry.pop(0)
		entry = [float(i) for i in entry]
		parsedData.append(entry)

	for entry in result:
		entry.pop(0)
		parsedResult.append(float(entry[0]))

	length = len(data)
	end = int(length * 0.8)

	parsedData = np.array(parsedData)

	mergedData = np.c_[parsedData, parsedResult]

	np.random.seed(0)
	randomize = np.arange(len(mergedData))
	np.random.shuffle(randomize)
	mergedData = mergedData[randomize]

	print mergedData

	(row, col) = mergedData.shape

	parsedResult = mergedData[:, col - 1]
	parsedData = np.delete(mergedData, [col - 1], axis=1)

	train_validate_data = parsedData[0:end]	
	train_validate_result = parsedResult[0:end]

	test_data = parsedData[end + 1 :]
	test_result = parsedResult[end + 1 :]

	# X data
	X = np.matrix(train_validate_data)

	# Y result
	Y = np.matrix(train_validate_result).transpose()

	Xtest = np.matrix(test_data)
	Ytest = np.matrix(test_result).transpose()

	counter = 0
	for y in Y:
		if y == 1:
			counter += 1

	print counter

	# alpha: step for gradientDescent
	alpha = 0.0005
	alphaList = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.005, 0.01]

	# Error threshold
	errorThres = 0.5
	errorThresList = [5, 1, 0.5, 0.1, 0.05]

	# for a in alphaList:
	# 	print a
	# 	(W, iteration, errorList) = logisticRegression(X, Y, a, errorThres)
	# 	plt.plot(iteration, errorList, markersize=5, label='$alpha = {a}$'.format(a=a))

	# for e in errorThresList:
	# 	print(e)
	# 	(W, iteration, errorList) = logisticRegression(X, Y, alpha, e)
	# 	plt.plot(iteration, errorList, markersize=5, label='$errThres = {e}$'.format(e=e))


	# plt.legend(loc='best')
	# plt.xlabel("Iterations")
	# plt.ylabel("Error")
	# plt.title("Error vs Nb of Iterations")
	# plt.show()

	# Testing
	#kFoldTesting(X, Y, alpha, errorThres)

	(trainingW, iter, errorList) = logisticRegression(X, Y, alpha, errorThres)

	

	test(Xtest, Ytest, trainingW)

main()
