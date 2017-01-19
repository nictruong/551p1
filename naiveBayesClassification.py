import csv
import numpy as np
import math


class NaiveBayesClassification:

    def load_csv(self):
        data = []
        with open('Project1_data.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)

        data = data[1:]

        output = []
        result = []

        data_by_id = {}
        for i in data:
            id = int(i[0])
            if id not in data_by_id:
                data_by_id[id] = []
            data_by_id[id].append(i)

        for id in data_by_id:
            individualData = data_by_id[id]
            ageSum = 0
            timeSum = 0
            rankSum = 0
            latestParticipation = 0
            numberOfParticipations = 0
            participatedIn2016 = False
            participatedIn2015 = False
            for individualEntry in individualData:
                year = int(individualEntry[7])
                if year == 2016:
                    participatedIn2016 = True
                    continue

                if year == 2015:
                    participatedIn2015 = True

                numberOfParticipations += 1
                ageSum += int(individualEntry[2])
                splitTime = individualEntry[5].split(":")
                timeSum += int(splitTime[0]) * 3600 + int(splitTime[1]) * 60 + int(splitTime[2])
                rankSum += int(individualEntry[4])
                participationYear = int(individualEntry[7])
                if participationYear > latestParticipation:
                    latestParticipation = participationYear

            if latestParticipation != 2016 and len(individualData) != 1:
                averageAge = float(ageSum) / float(len(individualData))
                sex = 0 if (individualData[0][3] == "F") else 1  # 0 = female 1 = male
                averageTime = float(timeSum) / float(len(individualData))  # in seconds
                averageRank = float(rankSum) / float(len(individualData))
                participatedIn2015_int = 1 if participatedIn2015 else 0
                participatedIn2016_int = 1 if participatedIn2016 else 0
                individualOutput = [id, averageAge, float(sex), averageTime, averageRank,
                                    float(numberOfParticipations),
                                    participatedIn2015_int]
                if numberOfParticipations < 20:
                    output.append(individualOutput)
                    result.append([id, participatedIn2016_int])

        return output, result

    def separate_data_by_class(self):
        separated = {}
        for i in range(len(self.dataset)):
            vector = self.dataset[i]
            #print vector
            if (vector[-1] not in separated):
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector[1:-1])
        return separated

    def compute_separated_statistics(self, separated):
        separated_statistics = {}
        for i in [0, 1]:
            X = np.array(separated[i])
            separated_statistics[i] = (X.mean(0), X.var(0))
        return separated_statistics

    def compute_probability(self, x, mean, variance):
        return 1/(np.sqrt(2*np.pi * variance)) * np.exp(-((x - mean) ** 2)/(2*variance))

    def compute_class_probability(self, features, for_class):
        total = 0
        for i in range(len(features)):
            total += math.log(self.compute_probability(features[i],
                              self.separated_statistics[for_class][0][i],
                              self.separated_statistics[for_class][1][i]))
            #print total
            #total *= 0.5 #class probability
        return total

    def __init__(self):
        print "Go!"
        self.dataset, self.result = self.load_csv()
        print "Step1"
        separated_data = self.separate_data_by_class()
        self.separated_statistics = self.compute_separated_statistics(separated_data)
        print self.separated_statistics
        nb_success = 0
        for entry in self.dataset:
            prob_0 = self.compute_class_probability(entry[1:-1], 0)
            prob_1 = self.compute_class_probability(entry[1:-1], 1)
            if prob_0 > prob_1:
                predicted_class = 0
            else:
                predicted_class = 1

            for i in self.result:
                if i[0] == entry[0]:
                    if predicted_class == i[1]:
                        nb_success += 1
        print nb_success
        print "Rate :"+str(nb_success / float(len(self.dataset)))

        #print separated_statistics

NaiveBayesClassification()