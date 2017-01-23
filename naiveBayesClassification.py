import csv
import numpy as np
import math
import random


class NaiveBayesClassification:

    def load_csv(self, prediction_year):
        data = []
        targetYear = prediction_year - 1
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

        data_by_id = self.remove_double_id(data_by_id)
        data_by_id = self.remove_entry_above(data_by_id, prediction_year)
        for id in data_by_id:
            for individualEntry in data_by_id[id]:
                year = int(individualEntry[7])
                if year > prediction_year:
                    data_by_id[id].remove(individualEntry)

        for id in data_by_id:
            individualData = data_by_id[id]
            ageSum = timeSum = rankSum = 0
            latestParticipation = numberOfParticipations = 0
            participatedInTargetYear = participatedInPredictionYear = False

            for individualEntry in individualData:
                year = int(individualEntry[7])
                if year > prediction_year:
                    continue

                if year == prediction_year:
                    participatedInPredictionYear = True
                    continue

                if year == targetYear:
                    participatedInTargetYear = True

                numberOfParticipations += 1
                ageSum += int(individualEntry[2])
                splitTime = individualEntry[5].split(":")
                timeSum += int(splitTime[0]) * 3600 + int(splitTime[1]) * 60 + int(splitTime[2])
                rankSum += int(individualEntry[4])
                participationYear = int(individualEntry[7])
                if participationYear > latestParticipation:
                    latestParticipation = participationYear

            if latestParticipation != prediction_year and numberOfParticipations > 1:
                averageAge = float(ageSum) / float(numberOfParticipations)
                sex = 0 if (individualData[0][3] == "F") else 1  # 0 = female 1 = male
                averageTime = float(timeSum) / float(numberOfParticipations)  # in seconds
                averageRank = float(rankSum) / float(numberOfParticipations)
                participatedInTargetYear = 1 if participatedInTargetYear else 0
                participatedInPredictionYear = 1 if participatedInPredictionYear else 0
                individualOutput = [id, sex, averageAge, averageTime, averageRank,
                                    float(numberOfParticipations),
                                    participatedInTargetYear]
                if numberOfParticipations < 20:  # "private" runner problem
                    output.append(individualOutput)
                    result.append([id, participatedInPredictionYear])

        return output, result

    def remove_double_id(self, data_by_id):
        bad_ids = set()
        for id in data_by_id:
            years = set()
            for individualEntry in data_by_id[id]:
                year = individualEntry[7]
                if year not in years:
                    years.add(year)
                else:
                    bad_ids.add(id)
        for id in bad_ids:
            del data_by_id[id]

        return data_by_id

    def remove_entry_above(self, data_by_id, year):
        for id in data_by_id:
            for individualEntry in data_by_id[id]:
                entry_year = int(individualEntry[7])
                if entry_year > year:
                    data_by_id[id].remove(individualEntry)
        return data_by_id

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
        total = math.log(len(self.separated_data_training[for_class]) / float(len(self.dataset)))
        for i in range(1, len(features)):
            total += math.log(self.compute_gaussian_probability(features[i],
                                                                self.separated_statistics[for_class][0][i],
                                                                self.separated_statistics[for_class][1][i]))
        # sex binary probability
        total += math.log(self.compute_binary_probability(0, features[0], for_class))
        return total

    def predict2017(self):
        self.dataset, result = self.load_csv(2017)

        # split training data set in two parts: one by class (0 = not participated in 2015, 1 = participated in 2016)
        self.separated_data_training = self.separate_data_by_class(self.dataset)

        # for each class compute statistics (mean and variance) for real values features
        self.separated_statistics = self.compute_separated_statistics(self.separated_data_training)

        # make a prediction for each id in the dataset
        with open("prediction_2017_bayes.csv", "wt") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "PredictionFor2017"])
            for entry in self.dataset:
                prob_0 = self.compute_class_probability(entry[1:-1], 0)
                prob_1 = self.compute_class_probability(entry[1:-1], 1)
                if prob_0 > prob_1:
                    predicted_class = 0
                else:
                    predicted_class = 1
                writer.writerow([entry[0], predicted_class])

    def predict_for(self, year):
        self.dataset, result = self.load_csv(year)

        # split training data set in two parts: one by class (0 = not participated in 2015, 1 = participated in 2016)
        self.separated_data_training = self.separate_data_by_class(self.dataset)

        # for each class compute statistics (mean and variance) for real values features
        self.separated_statistics = self.compute_separated_statistics(self.separated_data_training)

        # validation: How well can we predict for a given year?
        nb_success = nb_failure = 0
        for entry in self.dataset:
            prob_0 = self.compute_class_probability(entry[1:-1], 0)
            prob_1 = self.compute_class_probability(entry[1:-1], 1)
            if prob_0 > prob_1:
                predicted_class = 0
            else:
                predicted_class = 1

            for i in result:
                if i[0] == entry[0]:
                    if predicted_class == i[1]:
                        nb_success += 1
                    else:
                        nb_failure += 1
        print "Success Rate :" + str(nb_success / float(len(self.dataset)))

    def __init__(self):
        self.dataset = []
        self.separated_data_training = {}
        self.separated_statistics = {}



NaiveBayesClassification().predict_for(2007)
