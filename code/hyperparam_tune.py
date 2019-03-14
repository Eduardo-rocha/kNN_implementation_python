######################################################################################
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# February of 2019
# hyperparam_tune
######################################################################################


import json
import numpy as np 
import sys

def main(): # called at end of file
	# Receive arguments using sys ########################################################

	max_k = int(sys.argv[1])
	trainingSetPath = sys.argv[2]
	validationSetPath = sys.argv[3]
	testSetPath = sys.argv[4]

	#testSetPath = 'datasets/votes_test.json'
	#trainingSetPath = 'datasets/votes_train.json'

	# Load training set ##################################################################

	with open(trainingSetPath) as f:
	    trainSet = json.load(f)

	# Load test set
	with open(testSetPath) as f:
	    testSet = json.load(f)

	# Load validation set
	with open(validationSetPath) as f:
	    validationSet = json.load(f)

	# Tuning #############################################################################
	k_accuracies = [0 for j in range(max_k)]
	for k in range(1,max_k+1):
		k_accuracies[k-1] = kNNAccuracy(trainSet, validationSet, k)
		print(str(k)+","+str(k_accuracies[k-1]))

	best_k = np.argmax(k_accuracies)+1

	# Test set ###########################################################################

	# Merge training set with validation set
	trainSet["data"] = trainSet["data"]+validationSet["data"]
	test_accuracy = kNNAccuracy(trainSet, testSet, best_k)
	print(best_k)
	print(test_accuracy)
	
	return


######################################################################################
# Code from problem 1 used as function to calculate accuracies
######################################################################################
def kNNAccuracy(trainSet_f, testSet_f, k):
	features = testSet_f["metadata"]["features"][:-1]
	numberFeatures = len(features)

	trainingData = trainSet_f["data"]
	testData = testSet_f["data"]

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
		numberLabels = len(testSet_f["metadata"]["features"][-1][1])
		labelsCount = [0 for j in range(numberLabels)]
		for j in range(k):
			for l in range(numberLabels):
				label = testSet_f["metadata"]["features"][-1][1][l]
				if label == trainingData[kNN_indexes[j]][-1]:
					labelsCount[l] += 1
		#print(labelsCount)

		# mode of labels - first label if tie
		prediction = testSet_f["metadata"]["features"][-1][1][np.argmax(labelsCount)]
		correctLabel = testData[i][-1]
		#print(prediction)
		#print(correctLabel)

		# Compute Accuracy variables
		totalTests += 1
		if prediction == correctLabel: correctPredictions += 1

	# Accuracy of test set ###############################################################
	accuracy = correctPredictions/totalTests
	return(accuracy)


main()

