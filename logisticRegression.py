import numpy as np
import csv
from math import exp
from math import log

def findWeights( X, Y ):

	# transpose of X
	XT = X.transpose()

	# dot multiplication of XT * X
	XTX = np.dot(XT, X)

	# inverse of XTX
	XTXinv = np.linalg.inv(XTX)

	# dot of XT * Y
	XTY = np.dot(XT, Y)

	# dot of XTXinv * XTY to find weights
	W = np.dot(XTXinv, XTY)

	return W


# Class 4 Slide 15
def logistic ( WT, x ):

	WTx = np.dot(WT, x)

	return (1.0 / (1.0 + exp(-WTx)))


# Class 4 Slide 17
def error ( X, Y, W ):

	sum = 0.0

	WT = W.transpose()

	for x, y in zip(X, Y):
		sum += y * log(logisticFun(WT, x))
		try:
			sum += (1.0 - y) * log(1.0 - logisticFun(WT, x))
		except Exception, e:
			return -sum

	return -sum


# Class 4 Slide 17
def getNewW ( X, Y, W, alpha ):
	
	WT = W.transpose()

	sum = 0.0

	for x, y in zip(X, Y):
		sum += np.dot(x, y - logisticFun(WT, x))

	temp = alpha * sum

	return W + temp


def gradientDescentForLogReg( X, Y, W, alpha ):

	# transpose of W
	WT = W.transpose()

	newW = getNewW(X, Y, W, alpha)



	return newW


def logisticRegression( X, Y, alpha ):

	W = findWeights(X, Y)

	optimizedW = gradientDescentForLogReg(X, Y, W, alpha)	

	print(optimizedW)

	return optimizedW



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

data = data[1:500]	
result = result[1:500]

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
X = np.array(parsedData)

# Y result
Y = np.array(parsedResult)

# alpha: step for gradientDescent
alpha = 0.000000002

# Weights
W = logisticRegression(X, Y, alpha)