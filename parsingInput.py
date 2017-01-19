import csv


#########################################################################################################################
# This file currently completely removes points where individuals ONLY ran in 2016.
# It then collapses a person's data, calculating statistics WITHOUT their 2016 run if it exists.
# id, averageAge, sex, averageTime, averageRank, weightedAppearance, yearsOfParticipation, yearsSinceLastParticipation
#########################################################################################################################

ageDist = {
	'g-14': 0,
	'g15-19': 0,
	'g20-24': 0,
	'g25-29': 0,
	'g30-34': 0,
	'g35-39': 0,
	'g40-44': 0,
	'g45-49': 0,
	'g50-54': 0,
	'g55-59': 0,
	'g60-64': 0,
	'g65-69': 0,
	'g70-74': 0,
	'g75-79': 0,
	'g80-84': 0,
	'g85-89': 0,
	'g90-94': 0,
	'g95-': 0,
}


def main():
	data = []
	with open('Project1_data.csv','rt') as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        data.append(row)

	data = data[1:]

	aggregatedIndividuals = aggregateIndividuals(data)

	# update the age distribution curve
	for aggregatedIndividual in aggregatedIndividuals:
		for individualEntry in aggregatedIndividual:
			# Ignore 2016 entries, as they will be used for prediction
			if (int(individualEntry[7]) == 2016):
				participatedIn2016 = True
				continue
			else:
				updateAgeDistribution(int(individualEntry[2]))

	(collapsedIndividuals, result) = collapseIndividuals(aggregatedIndividuals)

	with open("parsedOutput.csv", "wt", newline='') as f:
		writer = csv.writer(f)
		writer.writerow([ "ID", "ageCategoryWeight", "sex", "averageTime", "averageRank", "weightedAppearance", "yearsOfParticipation", "yearsSinceLastParticipation" ])
		writer.writerows(collapsedIndividuals)

	with open("result.csv", "wt", newline='') as f:
	    writer = csv.writer(f)
	    writer.writerow([ "ID", "participatedIn2016" ])
	    writer.writerows(result)


# Aggregates each runner in a list
def aggregateIndividuals(data):

	i = 0
	dataLength = len(data);

	individualDatas = []

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

		individualDatas.append(individualData)

	return individualDatas

# Create parameters for each runner
def collapseIndividuals(individualDatas):

	output = []
	resultY = []


	for individualData in individualDatas:

		#find various stats
		ageSum = 0
		timeSum = 0
		rankSum = 0
		latestParticipation = 0
		weightedAppearanceSum = 0
		participatedIn2016 = False
		noEntries = len(individualData)

		for individualEntry in individualData:

			# Ignore 2016 entries, as they will be used for prediction
			if (int(individualEntry[7]) == 2016):
				participatedIn2016 = True
				continue
			else:
				# ageSum += int(individualEntry[2])

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

		if (participatedIn2016 and noEntries == 1):
			continue
		else:
			id = individualData[0][0]
			ageCategoryWeight = getAgeCategoryWeight(int(individualEntry[2]))
			# averageAge = float(ageSum) / float(noEntries)
			sex = 0 if (individualData[0][3] == "F") else 1 # 0 = female 1 = male
			averageTime =  float(timeSum) / float(noEntries) # in seconds
			averageRank = float(rankSum) / float(noEntries)
			weightedAppearance = weightedAppearanceSum
			yearsOfParticipation = noEntries - 1 if (participatedIn2016) else noEntries
			yearsSinceLastParticipation = 0 if (latestParticipation == 0) else 2016 - latestParticipation

			individualOutput = [ id, ageCategoryWeight, sex, averageTime, averageRank, weightedAppearance, yearsOfParticipation, yearsSinceLastParticipation ]

			output.append(individualOutput)

			if (participatedIn2016):
				resultY.append([ id, 1 ])
			else:
				resultY.append([ id, 0 ])

	return (output, resultY)

# Create an ageDist
def updateAgeDistribution(age):
	global ageDist

	if (age <= 14):
		ageDist['g-14'] += 1
	elif(age >= 15 and age <= 19):
		ageDist['g15-19'] += 1
	elif(age >= 20 and age <= 24):
		ageDist['g20-24'] += 1
	elif(age >= 25 and age <= 29):
		ageDist['g25-29'] += 1
	elif(age >= 30 and age <= 34):
		ageDist['g30-34'] += 1
	elif(age >= 35 and age <= 39):
		ageDist['g35-39'] += 1
	elif(age >= 40 and age <= 44):
		ageDist['g40-44'] += 1
	elif(age >= 45 and age <= 49):
		ageDist['g45-49'] += 1
	elif(age >= 50 and age <= 54):
		ageDist['g50-54'] += 1
	elif(age >= 55 and age <= 59):
		ageDist['g55-59'] += 1
	elif(age >= 60 and age <= 64):
		ageDist['g60-64'] += 1
	elif(age >= 65 and age <= 69):
		ageDist['g65-69'] += 1
	elif(age >= 70 and age <= 74):
		ageDist['g70-74'] += 1
	elif(age >= 75 and age <= 79):
		ageDist['g75-79'] += 1
	elif(age >= 80 and age <= 84):
		ageDist['g80-84'] += 1
	elif(age >= 85 and age <= 89):
		ageDist['g85-89'] += 1
	elif(age >= 90 and age <= 94):
		ageDist['g90-94'] += 1
	elif(age >= 95):
		ageDist['g95-'] += 1

def getAgeCategoryWeight(age):
	global ageDist

	weight = 0

	if (age <= 14):
		weight = ageDist['g-14']
	elif(age >= 15 and age <= 19):
		weight = ageDist['g15-19']
	elif(age >= 20 and age <= 24):
		weight = ageDist['g20-24']
	elif(age >= 25 and age <= 29):
		weight = ageDist['g25-29']
	elif(age >= 30 and age <= 34):
		weight = ageDist['g30-34']
	elif(age >= 35 and age <= 39):
		weight = ageDist['g35-39']
	elif(age >= 40 and age <= 44):
		weight = ageDist['g40-44']
	elif(age >= 45 and age <= 49):
		weight = ageDist['g45-49']
	elif(age >= 50 and age <= 54):
		weight = ageDist['g50-54']
	elif(age >= 55 and age <= 59):
		weight = ageDist['g55-59']
	elif(age >= 60 and age <= 64):
		weight = ageDist['g60-64']
	elif(age >= 65 and age <= 69):
		weight = ageDist['g65-69']
	elif(age >= 70 and age <= 74):
		weight = ageDist['g70-74']
	elif(age >= 75 and age <= 79):
		weight = ageDist['g75-79']
	elif(age >= 80 and age <= 84):
		weight = ageDist['g80-84']
	elif(age >= 85 and age <= 89):
		weight = ageDist['g85-89']
	elif(age >= 90 and age <= 94):
		weight = ageDist['g90-94']
	elif(age >= 95):
		weight = ageDist['g95-']

	return weight

main()
