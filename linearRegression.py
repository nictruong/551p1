'''
Created on Jan 19, 2017

@author: eanuama
'''

import csv
import numpy
import math
import time


#Predicts the value of Y with the help of X and W matrix
def predictY(X, W):
    Y = numpy.dot(X, W)
    
    return Y
    
    

#Calculates the average square error between the actual value and the predicted value
def calculateTrainingError(Ypredict, Y):
    resultDiff = numpy.subtract(Y, Ypredict)
    
    diffArray = numpy.squeeze(numpy.asarray(resultDiff))
    
    total = len(diffArray)    
    avgSquareError = 0
    
    for diff in diffArray:
        squareDiff = diff*diff
        avgSquareError = avgSquareError +  squareDiff
    
    avgSquareError = avgSquareError / total
    
    
    return avgSquareError
    
    


#Performs K fold cross validation
def kFoldCrossValidation():
    totalTrainingErrorAverage = 0
    totalValidationErrorAverage = 0
    #Each fold will have almost 6000 records    
    sizeOfPartition = 6000        
    dataInput = []
    output = []
    trainingErrorWMap = dict()
        
    
    
    with open('processedInput.csv','rt') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "ID":
                continue
            dataInput.append(row)
       
    
    
    
    with open('outputFile.csv','rt') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == "ID":
                continue
        
            output.append(row)
            
            
    
    totalInput = len(dataInput)
    #Calculating the number of folds
    KFold = int(math.ceil(float(totalInput)/sizeOfPartition))
    

    
    currentInputDataIndex = 0   
    endI = 0
    
    for no in range(KFold):        
        testData = []
        testResult = []
        data = []
        result = []
            
    
        startI = currentInputDataIndex            
        currentPartitionSizeInput = 0
        
        #Testing partition
        while currentInputDataIndex < totalInput:
                   
            ##################
            # TestInput  #
            ##################
            row = dataInput[currentInputDataIndex]
                          
            avgAge = float(row[1])
            avgTime = float(row[2])    
            totalApperance = float(row[4])
            lastAppearance = float(row[6])
            sex = float(row[5])
            bias = 1.0               
             
            lastTime = float(row[6 + int(row[6])])  
            customRow = [avgAge, avgTime, totalApperance, lastAppearance, lastTime, sex, bias]                
            testData.append(customRow)
                        
            ##################
            # TestOutput #
            ##################
            outputRow = output[currentInputDataIndex]
            
            actualTime = float(outputRow[1])                
            customRowOutput = [actualTime]
            testResult.append(customRowOutput)    
            
            
            currentInputDataIndex = currentInputDataIndex + 1
            endI = currentInputDataIndex            
            currentPartitionSizeInput = currentPartitionSizeInput + 1       
            
            if currentPartitionSizeInput == sizeOfPartition:
                break    
        
                

        #Training partition
        if startI == 0:
            print "StartC: " + str(startI) + ":" + str(endI)
            i = endI
            breakCondition = totalInput
            while i < breakCondition:
                #Training Input
                row = dataInput[i]            
                          
                avgAge = float(row[1])
                avgTime = float(row[2])
                totalApperance = float(row[4])
                lastAppearance = float(row[6])
                sex = float(row[5])
                bias = 1.0                
                
                lastTime = float(row[6 + int(row[6])])
                customRow = [avgAge, avgTime, totalApperance, lastAppearance, lastTime, sex, bias]                
                data.append(customRow)
                
                #Training Output
                rowOutput = output[i]            
                actualTime = float(rowOutput[1])                
                customRow = [actualTime]
                result.append(customRow)
                
                i = i + 1
                
                        
            
        elif endI == totalInput:
            print "EndC: " + str(startI) + ":" + str(endI)
            i = 0
            breakCondition = startI 
            while i < breakCondition:
                #Training Input                
                row = dataInput[i]
                
                
                avgAge = float(row[1])
                avgTime = float(row[2])
                totalApperance = float(row[4])
                lastAppearance = float(row[6])
                sex = float(row[5])
                bias = 1.0                
                
                lastTime = float(row[6 + int(row[6])])
                customRow = [avgAge, avgTime, totalApperance, lastAppearance, lastTime, sex, bias]                
                data.append(customRow)
                
                #Training Output
                rowOutput = output[i]            
                actualTime = float(rowOutput[1])                
                customRow = [actualTime]
                result.append(customRow)                
                
                i = i + 1                
                     
            
                                     
        else:
            print "Between: " + str(startI) + ":" + str(endI)
            
            i = 0
            breakCondition = startI
            while i < breakCondition:
                #Training Input                
                row = dataInput[i]
                            
                avgAge = float(row[1])
                avgTime = float(row[2])
                totalApperance = float(row[4])
                lastAppearance = float(row[6])
                sex = float(row[5])
                bias = 1.0                
                
                lastTime = float(row[6 + int(row[6])])                
                customRow = [avgAge, avgTime, totalApperance, lastAppearance, lastTime, sex, bias]                
                data.append(customRow)
                
                #Training Output
                rowOutput = output[i]            
                actualTime = float(rowOutput[1])                
                customRow = [actualTime]
                result.append(customRow)
                
                i = i + 1


            i = endI
            breakCondition = totalInput
            while i < breakCondition:
                #Training Input                
                row = dataInput[i]
                            
                avgAge = float(row[1])
                avgTime = float(row[2])
                totalApperance = float(row[4])
                lastAppearance = float(row[6])
                sex = float(row[5])
                bias = 1.0
                                
                lastTime = float(row[6 + int(row[6])])
                customRow = [avgAge, avgTime, totalApperance, lastAppearance, lastTime, sex, bias]                
                data.append(customRow)
                
                #Training Output
                rowOutput = output[i]            
                actualTime = float(rowOutput[1])                
                customRow = [actualTime]
                result.append(customRow)
                
                i = i + 1                
                
                        
        #Training                
        X, Y, W = calculateLeastSquareSolution(data, result)
        Ypredict = predictY(X, W)
        avgSqrError = calculateTrainingError(Ypredict, Y)
        print "Training: " + str(time.strftime("%H:%M:%S", time.gmtime(avgSqrError)))
        totalTrainingErrorAverage = avgSqrError
        trainingErrorWMap[avgSqrError] = W 
        
        
        #Testing                         
        X_validation = numpy.matrix(testData)
        Y_validation = numpy.matrix(testResult)        
        
        Y_validation_predict = predictY(X_validation, W)        
        avgSqrError = calculateTrainingError(Y_validation_predict, Y_validation)
        print  "Testing: " + str(time.strftime("%H:%M:%S", time.gmtime(avgSqrError)))
        totalValidationErrorAverage = avgSqrError
            
    
    totalTrainingErrorAverage = totalTrainingErrorAverage / KFold
    totalValidationErrorAverage = totalValidationErrorAverage / KFold
    
    print "Total Training Error Average:" + time.strftime("%H:%M:%S", time.gmtime(totalTrainingErrorAverage))    
    print "Total Validation Error Average:" + time.strftime("%H:%M:%S", time.gmtime(totalValidationErrorAverage))
    return trainingErrorWMap
        



#Calculates least square solution (closed form solution)
def calculateLeastSquareSolution(dataInput, dataOuput):
    X = numpy.matrix(dataInput)  
    Y = numpy.matrix(dataOuput)
    

    Xtr = numpy.transpose(X)
        
    Xtr_X = numpy.dot(Xtr,X)
    
    Xtr_X_Inverse = numpy.linalg.inv(Xtr_X)    
    
    Xtr_Y = numpy.dot(Xtr, Y)    
    
    Xtr_X_Inverse_dot_Xtr_Y = numpy.dot(Xtr_X_Inverse, Xtr_Y)
    
    W = Xtr_X_Inverse_dot_Xtr_Y
    
        
    return X, Y, W

        

#Prepares input data for 2017 predictions, i.e. X matrix
def calculateY2017():
    data = []
    runnerIdList = []
    
    with open('processedInput_All.csv','rt') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:            
            
            try:
                runnerId = row[0]                
                runnerIdList.append(int(runnerId))
                
                avgAge = float(row[1])
                avgTime = float(row[2])
                totalApperance = float(row[4])                
                lastAppearance = float(row[6])
                sex = float(row[5])
                bias = 1.0
                
                lastTime = float(row[6 + int(row[6])])                
                customRow = [avgAge, avgTime, totalApperance, lastAppearance, lastTime, sex, bias]
                
                data.append(customRow)
                
            except:
                continue
                
        
    X = numpy.matrix(data)   
    return runnerIdList, X
            


#Writes the prediction of 2017's marathon in a csv file,  it has runnerId and finish time mapping
def writeOutput(outputMap):
    with open("outputFile_2017.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow([ "ID", "2017_FinishTime" ])
    
        for key in outputMap.keys():
            data = outputMap[key]
            customRow = [key, data]                        
            writer.writerow(customRow)            

    print "Done"
    
    
    

def main():
    #Training the model and getting appropriate Weight vector
    trainingErrorWMap = kFoldCrossValidation()
    
    #Choosing the weight vector with least error
    trainingErrorList = trainingErrorWMap.keys()
    trainingErrorList = sorted(trainingErrorList, key =int, reverse =False)
    
    
    
    W = trainingErrorWMap[trainingErrorList[0]]
    runnerIdList, X = calculateY2017()
    
    #Calculating the finish time for 2017 marathon
    Y2017 = predictY(X, W)
    result2017 = numpy.squeeze(numpy.asarray(Y2017))
    
    outputMap = dict()
    i = 0
    for runnerId in runnerIdList:
        outputMap[runnerId] = time.strftime("%H:%M:%S", time.gmtime(result2017[i]))
        i = i + 1
        
    #Writing the result of 2017 marathon in a file    
    writeOutput(outputMap)    
    
    
        
    
if __name__ == '__main__':
    main()