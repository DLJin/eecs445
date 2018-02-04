# EECS 445 - Winter 2018
# Project 1 - project1.py

import pandas as pd
import numpy as np
import itertools
import string

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn import metrics
from matplotlib import pyplot as plt

from helper import *


def select_classifier(penalty='l2', c=1.0, degree=1, r=0.0, class_weight='balanced'):
	"""
	Return a linear svm classifier based on the given
	penalty function and regularization parameter c.
	"""
	# TODO: Optionally implement this helper function if you would like to
	# instantiate your SVM classifiers in a single function. You will need
	# to use the above parameters throughout the assignment.


def extract_dictionary(df):
	"""
	Reads a panda dataframe, and returns a dictionary of distinct words
	mapping from each distinct word to its index (ordered by when it was found).
	Input:
		df: dataframe/output of load_data()
	Returns:
		a dictionary of distinct words that maps each distinct word
		to a unique index corresponding to when it was first found while
		iterating over all words in each review in the dataframe df
	"""
	word_dict = {}
	counter = 0

	for review in df['content']:
		translator = review.maketrans(string.punctuation, ' '*len(string.punctuation))
		review = review.translate(translator).lower()
		#print(review)
		review_list = review.split()
		for word in review_list:
			if not word in word_dict:
				word_dict[word] = counter;
				counter = counter + 1
				
		#print("\n")

	#print(word_dict)

	return word_dict


def generate_feature_matrix(df, word_dict):
	"""
	Reads a dataframe and the dictionary of unique words
	to generate a matrix of {1, 0} feature vectors for each review.
	Use the word_dict to find the correct index to set to 1 for each place
	in the feature vector. The resulting feature matrix should be of
	dimension (number of reviews, number of words).
	Input:
		df: dataframe that has the ratings and labels
		word_list: dictionary of words mapping to indices
	Returns:
		a feature matrix of dimension (number of reviews, number of words)
	"""
	number_of_reviews = df.shape[0]
	number_of_words = len(word_dict)
	feature_matrix = np.zeros((number_of_reviews, number_of_words), dtype = int)
	# TODO: Implement this function

	
	keys = list(word_dict.keys())
	reviews = []

	for review in df['content']:
		translator = review.maketrans(string.punctuation, ' '*len(string.punctuation))
		review = review.translate(translator).lower()
		reviews.append(review)

	#print(reviews)

	#print("number_of_reviews: ", number_of_reviews)
	#print("number_of_words: ", number_of_words)
	#print("len(reviews): ", len(reviews))
	

	for review_num in range(0, number_of_reviews):
		for word in reviews[review_num].split():
			if word in word_dict:
				feature_matrix[review_num][word_dict[word]] = 1
		#for word_num in range(0, number_of_words):
		#	if keys[word_num] in reviews[review_num].split():
		#		feature_matrix[review_num][word_num] = 1
		#			
			

	#print(feature_matrix.shape)
	#print(feature_matrix)

	return feature_matrix


def cv_performance(clf, X, y, k=5, metric="accuracy"):
	"""
	Splits the data X and the labels y into k-folds and runs k-fold
	cross-validation: for each fold i in 1...k, trains a classifier on
	all the data except the ith fold, and tests on the ith fold.
	Calculates the k-fold cross-validation performance metric for classifier
	clf by averaging the performance across folds.
	Input:
		clf: an instance of SVC()
		X: (n,d) array of feature vectors, where n is the number of examples
		   and d is the number of features
		y: (n,) array of binary labels {1,-1}
		k: an int specifying the number of folds (default=5)
		metric: string specifying the performance metric (default='accuracy'
			 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
			 and 'specificity')
	Returns:
		average 'test' performance across the k folds as np.float64
	"""
	# TODO: Implement this function
	#HINT: You may find the StratifiedKFold from sklearn.model_selection
	#to be useful

	skf = StratifiedKFold(n_splits = k)
	skf.get_n_splits(X,y)
	#print("skf: ", skf)

	training_data = []
	testing_data = []	
	#Put the performance of the model on each fold in the scores array
	scores = []

	#train, test = skf.split(X,y)
	#print(len(train), len(test))
	for train_index, test_index in skf.split(X, y):
		#print("TRAIN:", train_index, "TEST:", test_index)
		#print("len(TRAIN):", len(train_index), "len(TEST):", len(test_index))
		training_data.append(train_index)
		testing_data.append(test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf.fit(X_train, y_train)
		#print(clf.predict(X_test))
		#print(performance(y_true = y_test, y_pred = clf.predict(X_test), metric = metric))
		if metric == 'auroc':
			scores.append(performance(y_true = y_test, y_pred = clf.decision_function(X_test), metric = metric))
		else:		
			scores.append(performance(y_true = y_test, y_pred = clf.predict(X_test), metric = metric))
		

	#print("k: ", k)
	#print("len(training_data): ", len(training_data))
	#print("len(testing_data): ", len(testing_data))

	

	#And return the average performance across all fold splits.
	return np.array(scores).mean()


def select_param_linear(X, y, k=5, metric="accuracy", C_range = [], penalty='l2'):
	"""
	Sweeps different settings for the hyperparameter of a linear-kernel SVM,
	calculating the k-fold CV performance for each setting on X, y.
	Input:
		X: (n,d) array of feature vectors, where n is the number of examples
		and d is the number of features
		y: (n,) array of binary labels {1,-1}
		k: int specifying the number of folds (default=5)
		metric: string specifying the performance metric (default='accuracy',
			 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
			 and 'specificity')
		C_range: an array with C values to be searched over
	Returns:
		The parameter value for a linear-kernel SVM that maximizes the
		average 5-fold CV performance.
	"""
	# TODO: Implement this function
	#HINT: You should be using your cv_performance function here
	#to evaluate the performance of each SVM

	cv_performance_results = []

	for c in C_range:
		clf = SVC(kernel = 'linear', C = c, class_weight = 'balanced')
		cv_performance_results.append(cv_performance(clf, X, y, k, metric))

	print("metric: ", metric)
	print("cv_performance_results for: ", cv_performance_results)

	return max(cv_performance_results)


def plot_weight(X,y,penalty,metric,C_range):
	"""
	Takes as input the training data X and labels y and plots the L0-norm
	(number of nonzero elements) of the coefficients learned by a classifier
	as a function of the C-values of the classifier.
	"""

	print("Plotting the number of nonzero entries of the parameter vector as a function of C")
	norm0 = []

	# TODO: Implement this part of the function
	#Here, for each value of c in C_range, you should
	#append to norm0 the L0-norm of the theta vector that is learned
	#when fitting an L2-penalty, degree=1 SVM to the data (X, y)

	for c in C_range:
		print("Starting new c")
		clf = SVC(kernel = 'linear', C = c, class_weight = 'balanced')
		clf.fit(X, y)
		coefs = 0
		#print(clf.coef_)
		for x in clf.coef_:
			print(x)
			print(x.shape)
			for i in x:
				if not i == 0:
					coefs = coefs + 1;
		norm0.append(coefs)

	print(norm0)


	#This code will plot your L0-norm as a function of c
	plt.plot(C_range, norm0)
	plt.xscale('log')
	plt.legend(['L0-norm'])
	plt.xlabel("Value of C")
	plt.ylabel("Norm of theta")
	plt.title('Norm-'+penalty+'_penalty.png')
	plt.savefig('Norm-'+penalty+'_penalty.png')
	plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
	"""
		Sweeps different settings for the hyperparameters of an quadratic-kernel SVM,
		calculating the k-fold CV performance for each setting on X, y.
		Input:
			X: (n,d) array of feature vectors, where n is the number of examples
			   and d is the number of features
			y: (n,) array of binary labels {1,-1}
			k: an int specifying the number of folds (default=5)
			metric: string specifying the performance metric (default='accuracy'
					 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
					 and 'specificity')
			parameter_values: a (num_param, 2)-sized array containing the
				parameter values to search over. The first column should
				represent the values for C, and the second column should
				represent the values for r. Each row of this array thus
				represents a pair of parameters to be tried together.
		Returns:
			The parameter value(s) for a quadratic-kernel SVM that maximize
			the average 5-fold CV performance
	"""
	# TODO: Implement this function
	# Hint: This will be very similar to select_param_linear, except
	# the type of SVM model you are using will be different...

	C_range = param_range
	R_range = param_range

	results = []

	for c in C_range:
		for r in R_range:
			clf = SVC(kernel = 'poly', degree = 2, C = c, coef0 = r, class_weight = 'balanced')
			results.append((c, r, cv_performance(clf, X, y, k, metric)))
		

	print("metric: ", metric)
	print("results: ", results)

	results.sort(key=lambda tup: tup[2] )

	print("results sorted: ", results)

	return results[len(results) - 1]


def performance(y_true, y_pred, metric="accuracy"):
	"""
	Calculates the performance metric as evaluated on the true labels
	y_true versus the predicted labels y_pred.
	Input:
		y_true: (n,) array containing known labels
		y_pred: (n,) array containing predicted scores
		metric: string specifying the performance metric (default='accuracy'
				 other options: 'f1-score', 'auroc', 'precision', 'sensitivity',
				 and 'specificity')
	Returns:
		the performance as an np.float64
	"""
	# TODO: Implement this function
	# This is an optional but very useful function to implement.
	# See the sklearn.metrics documentation for pointers on how to implement
	# the requested metrics.

	if metric == "accuracy":
		return metrics.accuracy_score(y_true, y_pred)
	elif metric == "f1":
		return metrics.f1_score(y_true, y_pred)
	elif metric == "auroc":
		return metrics.roc_auc_score(y_true, y_pred)
	elif metric == "precision":
		return metrics.precision_score(y_true, y_pred)
	elif metric == "specificity":
		#tn = 0
		#fp = 0
		#for x in range(0, len(y_true)):
		#	if y_true[x] == -1 and y_pred[x] == -1:
		#			tn = tn + 1
		#	elif y_true[x] == -1 and y_pred[x] == 1:
		#		fp = fp + 1
		#return tn/(tn+fp)
		#tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
		c_m = metrics.confusion_matrix(y_true, y_pred)
		#return tn/(tn+fp)
		return c_m[1][1]/(c_m[1][0] + c_m[1][1])
	elif metric == "sensitivity":
		#tp = 0
		#fn = 0
		#for x in range(0, len(y_true)):
		#	if y_true[x] == 1 and y_pred[x] == 1:
		#		tp = tp + 1
		#	elif y_true[x] == 1 and y_pred[x] == -1:
		#		fn = fn + 1
		#return tp/(tp+fn)
		#tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
		c_m = metrics.confusion_matrix(y_true, y_pred)
		#return tp/(tp+fn)
		return c_m[0][0]/(c_m[0][0] + c_m[0][1])
		


def main():
	# Read binary data
	# NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
	#	   IMPLEMENTING generate_feature_matrix AND extract_dictionary
	X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data()
	#IMB_features, IMB_labels = get_imbalanced_data(dictionary_binary)
	#IMB_test_features, IMB_test_labels = get_imbalanced_test(dictionary_binary)

	#print("X_train shape: ", X_train.shape)
	#print("X_train sum: ", np.sum(X_train))
	#print("X_train len: ", len(X_train))
	print("X_train avg: ", np.sum(X_train)/len(X_train))

	# TODO: Questions 2, 3, 4

	c_values = []
	for x in range(0, 7):
		c_values.append(pow(10, x - 3))
	print(c_values)

	#c_values = [0.01]

	#metrics = ["accuracy", "f1", "auroc", "precision", "sensitivity", "specificity"]
	#for metric in metrics:
	#	select_param_linear(X = X_train, y = Y_train, metric = metric, C_range = c_values)
	#for metric in metrics:
	#	plot_weight(X_train, Y_train, penalty = "l2", metric = metric, C_range = c_values)	

	
	#select_param_linear(X = X_train, y = Y_train, metric = "accuracy", C_range = c_values)
	#select_param_linear(X = X_train, y = Y_train, metric = "f1", C_range = c_values)
	#select_param_linear(X = X_train, y = Y_train, metric = "auroc", C_range = c_values)
	#select_param_linear(X = X_train, y = Y_train, metric = "precision", C_range = c_values)
	#select_param_linear(X = X_train, y = Y_train, metric = "sensitivity", C_range = c_values)
	#select_param_linear(X = X_train, y = Y_train, metric = "specificity", C_range = c_values)

	#plot_weight(X_train, Y_train, penalty = "l2", metric = "accuracy", C_range = c_values)

	#clf = SVC(kernel = 'linear', C = 0.1, class_weight = 'balanced')
	#clf.fit(X_train, Y_train)
	#theta = clf.coef_
	#print(len(theta[0]))
	#keys = list(dictionary_binary.keys())
	#theta_pairs = []

	#for x in range(0, len(theta[0])):
	#	theta_pairs.append((theta[0][x], keys[x]))
	#theta_pairs.sort(key=lambda tup: tup[0] )
	#print(theta_pairs[len(theta_pairs) - 1])
	#print(theta_pairs[len(theta_pairs) - 2])
	#print(theta_pairs[len(theta_pairs) - 3])
	#print(theta_pairs[len(theta_pairs) - 4])
	#print(theta_pairs[0])
	#print(theta_pairs[1])
	#print(theta_pairs[2])
	#print(theta_pairs[3])

	select_param_quadratic(X = X_train, y = Y_train, k = 5, metric = "auroc", param_range = c_values)

	
	
	

	# Read multiclass data
	# TODO: Question 5: Apply a classifier to heldout features, and then use
	#	   generate_challenge_labels to print the predicted labels
	#multiclass_features, multiclass_labels, multiclass_dictionary = get_multiclass_training_data()
	#heldout_features = get_heldout_reviews(multiclass_dictionary)


if __name__ == '__main__':
	main()
