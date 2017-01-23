import numpy as np
import csv
from math import exp
from math import log
import math
import random
import matplotlib.pyplot as plt

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
		print("error: " + str(error(validationX, validationY, trainingW)))

		success = 0
		errorV = 0
		comingGuess = 0
		notComingGuess = 0
		successComing = 0
		successNotComing = 0

		actuallyCame = 0
		actuallyDidntCome = 0

		trainingWT = np.transpose(trainingW)

		print(trainingWT)

		for x, y in zip(validationX, validationY):

			x = np.transpose(x)

			if y == 1:
				actuallyCame += 1
			elif y == 0:
				actuallyDidntCome += 1

			logFunV = logistic(trainingWT, x)

			if (logFunV > 0.5 and y == 1):
				successComing += 1

			if (logFunV <= 0.5 and y == 0):
				successNotComing += 1				

			if (logFunV > 0.5):
				comingGuess += 1

			if (logFunV < 0.5):
				notComingGuess += 1

			if ((logFunV > 0.5 and y == 1) or (logFunV <= 0.5 and y == 0)):
				success += 1

			if ((logFunV <= 0.5 and y == 1) or (logFunV > 0.5 and y == 0)):
				errorV += 1

		overallSuccessPercent = float(success) / float(len(validationX))
		overallErrorPercent = float(errorV) / float(len(validationX))

		print("overallSuccess%: " + str(overallSuccessPercent))
		print("overallError%: " + str(overallErrorPercent))
		print("overallSuccess: " + str(success))
		print("overallError: " + str(errorV))
		print("comingGuess: " + str(comingGuess))
		print("notComingGuess: " + str(notComingGuess))
		print("successComing: " + str(successComing))
		print("successNotComing: " + str(successNotComing))
		print("actuallyCame: " + str(actuallyCame))
		print("actuallyDidntCome: " + str(actuallyDidntCome))

		stats.append([k, overallSuccessPercent, overallErrorPercent, success, errorV, comingGuess, notComingGuess, successComing, successNotComing, actuallyCame, actuallyDidntCome])


	with open("stats.csv", "wt") as f:
		writer = csv.writer(f)
		writer.writerow(["k", "overallSuccessPercent", "overallErrorPercent", "success", "errorV", "comingGuess", "notComingGuess", "successComing", "successNotComing", "actuallyCame", "actuallyDidntCome"])
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
	alphaList = [0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.005, 0.01]

	# Error threshold
	errorThres = 0.5
	errorThresList = [5, 1, 0.5, 0.1, 0.05]

	# Weights
	# for a in alphaList:
	# 	print a
	# 	(W, iteration, errorList) = logisticRegression(X, Y, a, errorThres)
	# 	plt.plot(iteration, errorList, markersize=5, label='$alpha = {a}$'.format(a=a))

	# for e in errorThresList:
	# 	print(e)
	# 	(W, iteration, errorList) = logisticRegression(X, Y, alpha, e)
	# 	plt.plot(iteration, errorList, markersize=5, label='$errThres = {e}$'.format(e=e))


	# plt.legend(loc='best')
	# plt.show()

	# Testing
	kFoldTesting(X, Y, alpha, errorThres)

main()
