import csv
import numpy as np
import math
import random


class NaiveBayesClassification:

    def load_csv(self):
        data = []
        with open('Project1_data.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)

        data = data[1:]
        random.shuffle(data)
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
            ageSum = timeSum = rankSum = 0
            latestParticipation = numberOfParticipations = 0
            participatedIn2016 = participatedIn2015 = False
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
                individualOutput = [id, sex, averageAge, averageTime, averageRank,
                                    float(numberOfParticipations),
                                    participatedIn2015_int]
                if numberOfParticipations < 20: #"private" runner problem
                    output.append(individualOutput)
                    result.append([id, participatedIn2016_int])

        return output, result

    def separate_data_by_class(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if vector[-1] not in separated:
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector[1:-1])
        return separated

    def compute_separated_statistics(self, separated):
        separated_statistics = {}
        for i in [0, 1]:
            X = np.array(separated[i])
            separated_statistics[i] = (X.mean(0), X.var(0))
        return separated_statistics

    def compute_gaussian_probability(self, x, mean, variance):
        return 1/(np.sqrt(2*np.pi * variance)) * np.exp(-((x - mean) ** 2)/(2*variance))

    def compute_binary_probability(self, rank_feature, value, for_class):
        count = 0
        for i in self.separated_data_training[for_class]:
            if i[rank_feature] == value:
                count += 1
        return count / float(len(self.separated_data_training[for_class]))

    def compute_class_probability(self, features, for_class):
        total = math.log(len(self.separated_data_training[for_class]) / float(self.training_set_size))
        for i in range(1, len(features)):
            total += math.log(self.compute_gaussian_probability(features[i],
                                                                self.separated_statistics[for_class][0][i],
                                                                self.separated_statistics[for_class][1][i]))
        # sex binary probability
        total += math.log(self.compute_binary_probability(0, features[0], for_class))
        return total

    def __init__(self):
        # extract data from vcs (result corresponds to 2016 participation)
        self.dataset, self.result = self.load_csv()

        # split the dataset in two parts
        self.training_set_size = int(0.5*len(self.dataset))
        training_set = self.dataset[0:self.training_set_size]
        validation_set = self.dataset[self.training_set_size+1:]

        # split training data set in two parts: one by class (0 = not participated in 2015, 1 = participated in 2016)
        self.separated_data_training = self.separate_data_by_class(training_set)

        # for each class compute statistics (mean and variance) for real values features
        self.separated_statistics = self.compute_separated_statistics(self.separated_data_training)

        # validation test : how well can we predict 2016 participation from [2003 ... 2015] data?
        nb_success = nb_failure = 0
        for entry in validation_set:
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
                    else:
                        nb_failure += 1
        print nb_success
        print nb_failure
        print "Rate :"+str(nb_success / float(len(validation_set)))


NaiveBayesClassification()