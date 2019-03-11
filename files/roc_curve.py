#!/usr/bin/env python3.6
# chmod +x fileName.py

######################################################################################
# CS 760, Spring 2019
# HW 1
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# February of 2019
# Problem 4: roc_curve
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


# Standarizing numeric features ######################################################

# vectors with means and stddevs
# length = # of numeric features
means = np.sum(numTrainingData,0)/len(trainingData) # length = # of numeric features

stddev_diff = np.array(\
	[numTrainingData[j][:]-means for j in range(len(trainingData))])
stddev_diff_squared = np.power(stddev_diff, 2)
stddev = np.sqrt(np.sum(stddev_diff_squared,0)/len(trainingData)) 

# avoid division by 0 - convetion:
for i in range(len(stddev)):
	if stddev[i] == 0: 
		stddev[i] = 1 
		means[i]  = 0

# Standarize training data = (x-mean)/stddev

numTrainingDataStd = np.array(\
	[np.divide(numTrainingData[j][:]-means, stddev) for j in range(len(trainingData))])
numTestDataStd = np.array(\
	[np.divide(numTestData[j][:]-means, stddev) for j in range(len(testData))])

#print(numTrainingDataStd)
#print(numTestDataStd)

# Calculate distances ################################################################
# Find k-NN           ################################################################

totalTests = 0
correctPredictions = 0

confidence = [0 for j in range(len(numTestDataStd))]
yi = [0 for j in range(len(numTestDataStd))]

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
	distanceInstance_aux = np.copy(distanceInstance)
	kNN_indexes = [0 for j in range(k)]
	for j in range(k):
		kNN_indexes[j] = np.argmin(distanceInstance_aux)
		distanceInstance_aux[kNN_indexes[j]] = np.amax(distanceInstance_aux)+1
	#print(kNN_indexes)

	
	# calculate wn to neighboors
	epslon = 10**(-5)
	wn = np.power(np.power(\
		np.array([distanceInstance[kNN_indexes[j]] for j in range(k)])\
		, 2) + epslon, -1)
	yn = [trainingData[kNN_indexes[j]][-1] for j in range(k)]
	yn = [yn[f] == testSet["metadata"]["features"][-1][1][0] for f in range(len(yn))]

	confidence[i] = np.divide(np.sum(np.multiply(wn, yn)),np.sum(wn))



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
	yi[i] = correctLabel == testSet["metadata"]["features"][-1][1][0]


# Roc ###############################################################

# sorted indices
sortedIndex = [0 for j in range(len(confidence))]
confidence_aux = np.copy(confidence)
yi_aux = np.copy(yi)
for i in range(len(sortedIndex)):
	sortedIndex[i] = np.argmax(confidence_aux)
	confidence_aux[sortedIndex[i]] = -1





num_pos = np.sum(yi)
num_neg = len(yi) - num_pos

TP = 0
FP = 0
last_TP = 0

j = 0
for i in sortedIndex:
	confidence_aux[j] = confidence[i]
	yi_aux[j] = yi[i]
	j += 1


for i in range(len(sortedIndex)):
	if (i>0):
		if (confidence_aux[i]!=confidence_aux[i-1]) and ( yi_aux[i] == False) and (TP > last_TP):
			FPR = FP/num_neg
			TPR =TP/num_pos
			print(str(FPR)+","+str(TPR))
			last_TP = TP

	if yi_aux[i]:
		TP += 1
	else:
		FP += 1


FPR = FP/num_neg
TPR =TP/num_pos
print(str(FPR)+","+str(TPR))






