import numpy as np
import csv

data = []
result = []
with open('output.csv','rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

# with open('result.csv','rt') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         data.append(row)

data = data[1:20]	

# remove ID from all entries
# convert to float

parsedData = []

for entry in data:
	entry.pop(0)
	entry = [float(i) for i in entry]
	parsedData.append(entry)

# X
X = np.array(parsedData)

# transpose of X
XT = X.transpose()

# dot multiplication of XT * X
XTX = np.dot(XT, X)

# inverse of XTX
XTXinv = np.linalg.inv(XTX)




