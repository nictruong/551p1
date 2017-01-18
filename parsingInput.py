import csv


#########################################################################################################################
# This file currently completely removes points where individuals ONLY ran in 2016.
# It then collapses a person's data, calculating statistics WITHOUT their 2016 run if it exists.
# id, averageAge, sex, averageTime, averageRank, weightedAppearance, yearsOfParticipation, yearsSinceLastParticipation
#########################################################################################################################


data = []
with open('Project1_data.csv','rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)


data = data[1:]
dataLength = len(data);

i = 0
output = []
resultY = []

# iterate over all entries
while (i < dataLength):

	individualId = data[i][0]
	individualData = []

	begin = i

	# group entries with similar id
	for j in range(begin, dataLength):

		if (data[j][0] == individualId):
			individualData.append(data[j])
			i = j + 1 # i is now next entry
		else:
			i = j # i is now current entry since currentEntry was not part of the group
			break

	#find various stats
	ageSum = 0
	timeSum = 0
	rankSum = 0
	latestParticipation = 0
	weightedAppearanceSum = 0
	participatedIn2016 = False

	for individualEntry in individualData:

		# Ignore 2016 entries, as they will be used for prediction
		if (int(individualEntry[7]) == 2016):
			participatedIn2016 = True
			continue
		else:
			ageSum += int(individualEntry[2])

			splitTime = individualEntry[5].split(":")
			timeSum += int(splitTime[0]) * 3600 + int(splitTime[1]) * 60 + int(splitTime[2])

			rankSum += int(individualEntry[4])

			participationYear = int(individualEntry[7])

			if (latestParticipation < participationYear):
				latestParticipation = participationYear

			if (participationYear >= 2003 and participationYear < 2006):
				weightedAppearanceSum += 1
			elif(participationYear >= 2006 and participationYear < 2009):
				weightedAppearanceSum += 2
			elif(participationYear >= 2009 and participationYear < 2013):
				weightedAppearanceSum += 3
			elif(participationYear >= 2013 and participationYear <= 2016):
				weightedAppearanceSum += 4

	noEntries = len(individualData)

	if (participatedIn2016 and noEntries == 1):
		continue
	else:
		id = individualData[0][0]
		averageAge = float(ageSum) / float(noEntries)
		sex = 0 if (individualData[0][3] == "F") else 1 # 0 = female 1 = male
		averageTime =  float(timeSum) / float(noEntries) # in seconds
		averageRank = float(rankSum) / float(noEntries)
		weightedAppearance = weightedAppearanceSum
		yearsOfParticipation = noEntries - 1 if (participatedIn2016) else noEntries
		yearsSinceLastParticipation = 0 if (latestParticipation == 0) else 2016 - latestParticipation

		individualOutput = [ id, averageAge, sex, averageTime, averageRank, weightedAppearance, yearsOfParticipation, yearsSinceLastParticipation ]

		output.append(individualOutput)

		if (participatedIn2016):
			resultY.append([ id, 1 ])
		else:
			resultY.append([ id, 0 ])

with open("parsedOutput.csv", "wt") as f:
    writer = csv.writer(f)
    writer.writerow([ "ID", "averageAge", "sex", "averageTime", "averageRank", "weightedAppearance", "yearsOfParticipation", "yearsSinceLastParticipation" ])
    writer.writerows(output)

with open("result.csv", "wt") as f:
    writer = csv.writer(f)
    writer.writerow([ "ID", "participatedIn2016" ])
    writer.writerows(resultY)
