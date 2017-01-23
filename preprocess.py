'''
Created on Jan 18, 2017

@author: eanuama
'''

import csv





#This creates new column for the input data
def prepareInput(outputYear):
    #Stores id and other data mapping
    dataMap = {} 
    
    #Stores id and finishTime mapping for 2016
    outputMap = {}
    
    
    ageHelper = dict()
    
    
    with open('Project1_data.csv','rt') as csvfile:
        reader = csv.reader(csvfile)        
        
        for row in reader:          
               
            runnerId = None   
            try:
                runnerId = int(row[0])
            except:
                continue            
      
            
            age = int(row[2])       
            sex = row[3]                         
        
            #converting time into seconds
            time = row[5].replace(" ","")            
            timeComponents = time.split(":")      
            totalTimeInSeconds = float(timeComponents[0])*60*60 + int(timeComponents[1])*60 + int(timeComponents[2])            
            
            pace = row[6].replace(" ","")
            paceComponents = pace.split(":")          
            totalPaceInSeconds = float(paceComponents[0])*60 + int(paceComponents[1])
            
            totalDistance = totalTimeInSeconds/totalPaceInSeconds
            
            #Converting half marathon into full marathon            
            if totalDistance < 26:
                totalTimeInSeconds = 2.3 * totalTimeInSeconds
                totalPaceInSeconds = totalTimeInSeconds / 26
                  
            year = int(row[7])            
            
            
            if year == outputYear:
                outputMap[runnerId] = [runnerId, totalTimeInSeconds]
            else:                
                if runnerId in dataMap:
                    data = dataMap[runnerId]
                                   
                    totalAvgTime = data[2]
                    totalAvgPace = data[3]
                    
                    #incrementing the number of apperance by 1
                    totalApperanceInMarathon = data[4] + 1   
                    totalPreviousAppearance = data[4]            
                    
                    if totalApperanceInMarathon == 1:
                        totalAvgTime = (float(totalAvgTime) + totalTimeInSeconds)/totalApperanceInMarathon
                        totalAvgPace = (float(totalAvgPace) + totalPaceInSeconds)/totalApperanceInMarathon
                    else:
                        totalAvgTime = (float(totalAvgTime)*totalPreviousAppearance + totalTimeInSeconds)/totalApperanceInMarathon
                        totalAvgPace = (float(totalAvgPace)*totalPreviousAppearance + totalPaceInSeconds)/totalApperanceInMarathon
                        
                        
                    
                    data[2] = totalAvgTime
                    data[3] = totalAvgPace
                    data[4] = totalApperanceInMarathon
                    
                    yearIndex = year - 1996
                    data[yearIndex] = totalTimeInSeconds
                    
                    dataMap[runnerId] = data                  
                    
                    #performs age corrections in the data set                    
                    ageCorrectionData = ageHelper[runnerId]
                    
                    if ageCorrectionData[0] > year:
                        ageHelper[runnerId] = [year, age]
                    
                     
                    
                else:
                    data = []
                    if outputYear == 2016:
                        data = [None]*20
                    elif outputYear == 2017:
                        data = [None]*21    
                    
                    data[0] = runnerId                    
                    
                    data[2] = totalTimeInSeconds
                    data[3] = totalPaceInSeconds
                    data[4] = 1
                    
                    if sex == "M":
                        data[5] = 1
                    else:
                        data[5] = 2                        
                    
                    yearIndex = year - 1996 
                    
                    #This stores the finish time for a give year
                    data[yearIndex] = totalTimeInSeconds                    
                                        
                    dataMap[runnerId] = data              
                    
                    ageData = [year, age]
                    ageHelper[runnerId] = ageData              
                    
                    
            

            
    resultMap = addMissingData(dataMap, outputMap, ageHelper, outputYear)
    
     
        
    writeProcessedInput(dataMap, outputYear)
    if outputYear == 2016:
        writeOutput(resultMap)
    
    
    

    





def addMissingData(dataMap, outputMap, ageHelper, outputYear):
    
    outputMapBuffer = dict()
    #print ageHelper
    
    
    for key in dataMap.keys():
              
        data = dataMap[key]    
        yearData = data[7:]        
        
        index = 0        
        lastAppearanceWeight = 1
        
        for temp in yearData:            
            if temp is not None:
                lastAppearanceWeight = index + 1
                
            index = index  + 1
        
        #adding appearance weight
        data[6] = lastAppearanceWeight
         
        #calculating average finish time, adding it to no appearance year
        filterNoneData = [x for x in yearData if x is not None]
        avgTime = float(sum(filterNoneData))/len(filterNoneData)        
        newYearData = [avgTime if val is None else val for val in yearData]   
        data[7:] = newYearData 
        
        runnerId = data[0]        
        
        #calculating the average age
        ageData = ageHelper[runnerId]
        year = ageData[0]
        age = ageData[1]
        
        
        totalYears = 0
        lowerBound = year
        lAge = age 
        while(lowerBound >= 2003):            
            totalYears = totalYears + lAge
            lAge = lAge - 1
            lowerBound = lowerBound - 1
            
        
        upperBound = year + 1 
        hAge = age + 1
        while(upperBound <= (outputYear - 1)):
            totalYears =  totalYears + hAge
            
            hAge = hAge + 1
            upperBound = upperBound + 1
            
            
        totalNumberOfYears = outputYear  - 2003
        avgAge = float(totalYears)/totalNumberOfYears
        
        
        
        data[1] =  avgAge
        dataMap[key] = data
             
        
        
        
        #If the runner has not appeared in 2016's marathon, then put avgtime as the finish time for the 2016 
        
        if runnerId not in outputMap:
            outputMap[runnerId] = [runnerId, avgTime]
        
        
        
        outputMapBuffer[key] = outputMap[key]
        
    
    return outputMapBuffer    
            
            


            
def writeProcessedInput(dataMap, outputYear):
    fileName = ""
    
    #Two input files are generated, one is till 2015 and other is till 2016
    if outputYear ==  2016:
        fileName = "processedInput.csv"
    elif outputYear ==  2017:
        fileName = "processedInput_All.csv"    
    
    with open(fileName, "wb") as f:
        writer = csv.writer(f)
        
        #if output year is 2016, put else condition for the year 2017
        if outputYear == 2016:
            writer.writerow([ "ID", "averageAge", "averageTime", "averagePace", "totalAppearance", "sex", "LastAppearanceWeight", "2003_FinishTime", "2004_FinishTime", \
                         "2005_FinishTime", "2006_FinishTime","2007_FinishTime","2008_FinishTime","2009_FinishTime","2010_FinishTime", \
                         "2011_FinishTime", "2012_FinishTime","2013_FinishTime","2014_FinishTime","2015_FinishTime"])
        elif outputYear == 2017:    
            writer.writerow([ "ID", "averageAge", "averageTime", "averagePace", "totalAppearance", "sex", "LastAppearanceWeight", "2003_FinishTime", "2004_FinishTime", \
                         "2005_FinishTime", "2006_FinishTime","2007_FinishTime","2008_FinishTime","2009_FinishTime","2010_FinishTime", \
                         "2011_FinishTime", "2012_FinishTime","2013_FinishTime","2014_FinishTime","2015_FinishTime", "2016_FinishTime"])
        
        for key in dataMap.keys():
            data = dataMap[key]
            
                       
            writer.writerow(data)
            





def writeOutput(outputMap):
    with open("outputFile.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow([ "ID", "2016_FinishTime" ])
    
        for key in outputMap.keys():
            data = outputMap[key]                        
            writer.writerow(data)            

    print "Done"
    
    
    
    
    
if __name__ == '__main__':
    prepareInput(2016)
    prepareInput(2017)