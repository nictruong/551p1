import csv

data = []
with open('Project1_data.csv','rt') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)


data = data[1:]
dataLength = len(data);

i = 0
output = []

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

	for individualEntry in individualData:
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

	id = individualData[0][0]
	averageAge = ageSum / noEntries
	sex = 0 if (individualData[0][3] == "F") else 1 # 0 = female 1 = male
	averageTime =  timeSum / noEntries # in seconds
	averageRank = rankSum / noEntries
	averageWeightedAppearance = weightedAppearanceSum / noEntries
	yearsOfParticipation = noEntries
	yearsSinceLastParticipation = 2017 - latestParticipation

	individualOutput = [ id, averageAge, sex, averageTime, averageRank, averageWeightedAppearance, yearsOfParticipation, yearsSinceLastParticipation ]

	output.append(individualOutput)

with open("output.csv", "wt") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "averageAge", "sex", "averageTime", "averageRank", "averageWeightedAppearance", "yearsOfParticipation", "yearsSinceLastParticipation"])
    writer.writerows(output)
