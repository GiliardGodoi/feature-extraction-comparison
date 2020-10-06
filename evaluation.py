#######################################
#######################################
# -*- coding: utf-8 -*-

#######################################
#######################################

import operator
# import sklearn
import pandas as pd
import warnings
import numpy as np
import argparse
# import catboost
import sys
warnings.filterwarnings("ignore")
from catboost import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
# from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
# from imblearn.ensemble import EasyEnsemble
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt    
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, make_scorer
from imblearn.metrics import geometric_mean_score


def preprocessing(dataset):
	global train, test, train_labels, test_labels
	dataset = pd.read_csv(data)
	labels = dataset['label']
	features = dataset[dataset.columns[1:25]]
	print(features)
	print(len(labels))
	# print(dataset.describe())
	train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.3,
                                                          random_state=12,
                                                          stratify=labels)
	sc = MinMaxScaler(feature_range=(0, 1))
	train = sc.fit_transform(train)
	test = sc.transform(test)
	# print(train)
	# print(test)
	# features = pd.DataFrame(features))
	# sm = ClusterCentroids(random_state=0)
	# sm = SMOTE(random_state=12)
	# sm = EditedNearestNeighbours()
	# sm = RepeatedEditedNearestNeighbours()
	# sm = AllKNN()
	# smt = RandomUnderSampler(random_state=2)
	# sm = RandomUnderSampler()
	# sm = TomekLinks(random_state=42)
	# sm = EasyEnsemble(random_state=0, n_subsets=10)
	# train, train_labels = sm.fit_sample(train, train_labels)
	# test, test_labels = smt.fit_sample(test, test_labels)
	# sc = Normalizer().fit(train)
	# sc = StandardScaler().fit(train)
	# sc = Binarizer(threshold=0.0).fit(train)
	# sc.fit(train)
	# train_normalize = sc.transform(train)
	return


def header(foutput):
	file = open(foutput, 'a')
	file.write("qParameter,Classifier,ACC,std_ACC,SE,std_SE,F1,std_F1,AUC,std_AUC,BACC,std_BACC,kappa,std_kappa,gmean,std_gmean")
	file.write("\n")
	return
	
	
def save_measures(classifier, foutput, scores):
	file = open(foutput, 'a')
	file.write("%s,%s,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f" % (i, classifier, scores['test_ACC'].mean(), 
	+ scores['test_ACC'].std(), scores['test_recall'].mean(), scores['test_recall'].std(), 
	+ scores['test_f1'].mean(), scores['test_f1'].std(), 
	+ scores['test_roc_auc'].mean(), scores['test_roc_auc'].std(),
	+ scores['test_ACC_B'].mean(), scores['test_ACC_B'].std(),
	+ scores['test_kappa'].mean(), scores['test_kappa'].std(),
	+ scores['test_gmean'].mean(), scores['test_gmean'].std()))
	file.write("\n")
	return


def evaluate_model_holdout(classifier, model, finput):
	df = pd.read_csv(finput)	
	labels = df.iloc[:, -1]
	features = df[df.columns[1:(len(df.columns) - 1)]]
	print(features)
	print(labels)
	train, test, train_labels, test_labels = train_test_split(features,
                                                          labels,
                                                          test_size=0.3,
                                                          random_state=12,
                                                          stratify=labels)
	sc = MinMaxScaler(feature_range=(0, 1))
	train = sc.fit_transform(train)
	print(train)
	test = sc.transform(test)
	# print(test)
	print(test_labels)
	clf = model
	model = clf.fit(train, train_labels)
	preds = clf.predict(test)
	accu = accuracy_score(test_labels, preds)
	recall = recall_score(test_labels, preds)
	f1 = f1_score(test_labels, preds)
	auc = roc_auc_score(test_labels, preds)
	matriz = (pd.crosstab(test_labels, preds, rownames=["REAL"], colnames=["PREDITO"], margins=True))
	print("Classificador: %s" % (classifier))
	print("Predições %s" % (preds))
	print("Acurácia Teste: %s" % (accu))
	print("Recall: %s" % (recall))
	print("F1: %s" % (f1))
	print("AUC: %s" % (auc))
	print("%s" % (matriz))
	return


def evaluate_model_cross(classifier, model, finput):
	#####################################
	header = pd.read_csv(finput, header=None, nrows=1)
	if header.iloc[0, 0] == "nameseq":
		df = pd.read_csv(finput)
	else:
		df = pd.read_csv(finput, header=None)
	X = df[df.columns[1:(len(df.columns) - 1)]]
	print(X)
	# X = df.iloc[:, 1:-1]
	y = df.iloc[:, -1]
	# y = df['label']
	print(y)
	#####################################
	pipe = Pipeline(steps=[
		('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),
		('clf', model)])
	scoring = {'ACC': 'accuracy', 'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc', 'ACC_B': 'balanced_accuracy', 'kappa': make_scorer(cohen_kappa_score), 'gmean': make_scorer(geometric_mean_score)}
	kfold = KFold(n_splits=10, shuffle=True, random_state=42)
	scores = cross_validate(pipe, X, y, cv=kfold, scoring=scoring)
	save_measures(classifier, foutput, scores)
	return



##########################################################################
##########################################################################
if __name__ == "__main__":
	print("\n")
	print("###################################################################################")
	print("#####################   Arguments: -i input -o output -l    #######################")
	print("##########              Author: Robson Parmezan Bonidia                 ###########")
	print("###################################################################################")
	print("\n")
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='csv format file, E.g., dataset.csv')
	parser.add_argument('-o', '--output', help='CSV format file, E.g., test.csv')
    # parser.add_argument('-k', '--kmer', help='Range of k-mer, E.g., 1-mer (1) or 2-mer (1, 2) ...')
    # parser.add_argument('-e', '--entropy', help='Type of Entropy, E.g., Shannon or Tsallis')
    # parser.add_argument('-q', '--parameter', help='Tsallis - q parameter')
	args = parser.parse_args()
	finput = str(args.input)
	foutput = str(args.output)
	experiments = { 
		"GaussianNB" : GaussianNB(),
		"DecisionTree" : DecisionTreeClassifier(criterion='gini', max_depth=2, max_leaf_nodes=None, random_state=63),
		"GradientBoosting" : GradientBoostingClassifier(n_estimators=400, learning_rate=3.0, max_depth=1, random_state=63),
		"RandomForest" : RandomForestClassifier(random_state=63, n_estimators=100),
		"LogisticRegression" : LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5),
		"SVM" : svm.SVC(gamma = 'scale', kernel = 'poly', degree = 5, coef0 = 0.1, random_state = 63),
		"Bagging" : BaggingClassifier(random_state = 63),
		"KNN" : KNeighborsClassifier(),
		"Adaboost" : AdaBoostClassifier(random_state = 63),
		"MLP" : MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), learning_rate_init=0.001, random_state=63),
		"Catboost" : CatBoostClassifier(iterations=100, random_seed=63, logging_level = 'Silent'),
		"HistGradientBoosting" : HistGradientBoostingClassifier(random_state=63)
		}
	# foutput = "results_Covid1.csv"
	header(foutput)
	for i in np.arange(6.0, 6.1, 1.0):
		i = round(i, 1)
		print("Round: %s" % (i))
		# finput = "COVID-19/q/" + str(i) + ".csv"
		# finput = "train_other_viruses.csv"
		print(finput)
		for classifier, model in experiments.items():
			# print(classifier)
			# print(model)
			evaluate_model_cross(classifier, model, finput)
			# evaluate_model_holdout(classifier, model, finput)
##########################################################################
##########################################################################