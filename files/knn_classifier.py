#!/usr/bin/env python3.6
# chmod +x fileName.py

######################################################################################
# CS 760, Spring 2019
# HW 1
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# February of 2019
# Problem 1: knn_classifier
######################################################################################


import json
import numpy as np 
import sys


# Receive arguments using sys ########################################################

k = int(sys.argv[1])
trainingSetPath = sys.argv[2]
testSetPath = sys.argv[3]

#testSetPath = 'datasets/votes_test.json'
#trainingSetPath = 'datasets/votes_train.json'

# Load training set ##################################################################

with open(trainingSetPath) as f:
    trainSet = json.load(f)

# Load test set
with open(testSetPath) as f:
    testSet = json.load(f)

features = testSet["metadata"]["features"][:-1]
numberFeatures = len(features)

trainingData = trainSet["data"]
testData = testSet["data"]

# Separate numeric and categorical features ##########################################
# Allow vectorized calculations when possible

numFeatures = [f for f in features if f[1] == "numeric"]
catFeatures = [f for f in features if f[1] != "numeric"]
#print(numFeatures)
#print(catFeatures)

numTrainingData =np.array(\
	[[instance[f] for f in range(numberFeatures) if features[f][1] == "numeric"]\
	for instance in trainingData])
catTrainingData =np.array(\
	[[instance[f] for f in range(numberFeatures) if features[f][1] != "numeric"]\
	for instance in trainingData])

numTestData =np.array(\
	[[instance[f] for f in range(numberFeatures) if features[f][1] == "numeric"]\
	for instance in testData])
catTestData =np.array(\
	[[instance[f] for f in range(numberFeatures) if features[f][1] != "numeric"]\
	for instance in testData])

#print(numTrainingData)
#print(catTrainingData)
#print(numTestData)
#print(catTestData)

# Standarizing numeric features ######################################################

# vectors with means and stddevs
# length = # of numeric features
means = np.sum(numTrainingData,0)/len(trainingData) # length = # of numeric features
#print(means)

stddev_diff = np.array(\
	[numTrainingData[j][:]-means for j in range(len(trainingData))])
stddev_diff_squared = np.power(stddev_diff, 2)
stddev = np.sqrt(np.sum(stddev_diff_squared,0)/len(trainingData)) 

# avoid division by 0 - convetion:
for i in range(len(stddev)):
	if stddev[i] == 0: 
		stddev[i] = 1 
		means[i]  = 0

#print(stddev_diff)
#print(stddev_diff_squared)
#print(stddev)

# Standarize training data = (x-mean)/stddev

numTrainingDataStd = np.array(\
	[np.divide(numTrainingData[j][:]-means, stddev) for j in range(len(trainingData))])
numTestDataStd = np.array(\
	[np.divide(numTestData[j][:]-means, stddev) for j in range(len(testData))])

#print(numTrainingDataStd)
#print(numTestDataStd)

# Calculate distances ################################################################
# Find k-NN           ################################################################
# Predict label       ################################################################

totalTests = 0
correctPredictions = 0

for i in range(len(numTestDataStd)):
	# Numeric features - Manhattan distance
	testInstance = numTestDataStd[i]
	numDistancesInstance = np.array(\
		[np.sum(np.absolute(numTrainingDataStd[j][:]-testInstance), 0)\
		for j in range(len(trainingData))])
	#print(numDistancesInstance)

	# Categorical features - Hamming distance
	testInstance = catTestData[i]
	catDistancesInstance = np.array(\
		[np.sum([catTrainingData[j][f] != testInstance[f] for f in range(len(catFeatures))],0)\
		for j in range(len(trainingData))])
	#print(catDistancesInstance)

	# Total distance
	distanceInstance = np.add(numDistancesInstance, catDistancesInstance)
	#print(distanceInstance)

	# Find k-NN
	kNN_indexes = [0 for j in range(k)]
	for j in range(k):
		kNN_indexes[j] = np.argmin(distanceInstance)
		distanceInstance[kNN_indexes[j]] = np.amax(distanceInstance)+1
	#print(kNN_indexes)

	# Count labels in k-NN
	numberLabels = len(testSet["metadata"]["features"][-1][1])
	labelsCount = [0 for j in range(numberLabels)]
	for j in range(k):
		for l in range(numberLabels):
			label = testSet["metadata"]["features"][-1][1][l]
			if label == trainingData[kNN_indexes[j]][-1]:
				labelsCount[l] += 1
	#print(labelsCount)

	# mode of labels - first label if tie
	prediction = testSet["metadata"]["features"][-1][1][np.argmax(labelsCount)]
	correctLabel = testData[i][-1]
	#print(prediction)
	#print(correctLabel)

	# Compute Accuracy variables
	totalTests += 1
	if prediction == correctLabel: correctPredictions += 1

	# Print output
	for l in range(numberLabels):
		print(str(labelsCount[l])+",", end='')
	print(prediction)

# Accuracy of test set ###############################################################

accuracy = correctPredictions/totalTests
#print('\nTest set accuracy: ',accuracy,'\n')














































