# -*- coding: utf-8 -*-

import argparse
import csv
import operator
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# from catboost import *
from catboost import CatBoostClassifier

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import MLPClassifier

from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_score
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks

from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt

# def preprocessing(dataset):
# 	global train, test, train_labels, test_labels
# 	dataset = pd.read_csv(data)
# 	labels = dataset['label']
# 	features = dataset[dataset.columns[1:25]]
# 	print(features)
# 	print(len(labels))
# 	# print(dataset.describe())
# 	train, test, train_labels, test_labels = train_test_split(features,
#                                                           labels,
#                                                           test_size=0.3,
#                                                           random_state=12,
#                                                           stratify=labels)
# 	sc = MinMaxScaler(feature_range=(0, 1))
# 	train = sc.fit_transform(train)
# 	test = sc.transform(test)
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
# return

def read_dataset(filename):

    header = pd.read_csv(filename, header=None, nrows=1)

    if header.iloc[0, 0] == "nameseq":
        frame = pd.read_csv(finput)
    else:
        frame = pd.read_csv(finput, header=None)

    return frame

# def header(foutput):

# 	if not os.path.exists(foutput):
# 		file = open(foutput, 'a')
# 		file.write("qParameter,Classifier,ACC,std_ACC,SE,std_SE,F1,std_F1,AUC,std_AUC,BACC,std_BACC,kappa,std_kappa,gmean,std_gmean")
# 		file.write("\n")
# 	return


# def save_measures(classifier, foutput, scores):
# 	file = open(foutput, 'a')
# 	file.write("%s,%s,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f,%0.4f,%0.2f" % (i, classifier, scores['test_ACC'].mean(),
# 	+ scores['test_ACC'].std(), scores['test_recall'].mean(), scores['test_recall'].std(),
# 	+ scores['test_f1'].mean(), scores['test_f1'].std(),
# 	+ scores['test_roc_auc'].mean(), scores['test_roc_auc'].std(),
# 	+ scores['test_ACC_B'].mean(), scores['test_ACC_B'].std(),
# 	+ scores['test_kappa'].mean(), scores['test_kappa'].std(),
# 	+ scores['test_gmean'].mean(), scores['test_gmean'].std()))
# 	file.write("\n")
# 	return

def save_measures(results, f_output):

    header = ["qParameter",
              "Classifier",
              "ACC", "std_ACC",
              "SE", "std_SE",
              "F1", "std_F1",
              "AUC", "std_AUC",
              "BACC", "std_BACC",
              "kappa", "std_kappa",
              "gmean", "std_gmean"]

    file_exists = os.path.exists(f_output)

    with open(f_output, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        print("\t\tsaving...")
        writer.writerow(results)


def calculate_results(index, classifier, scores):
    results = [
        index,
        classifier,
        round(scores['test_ACC'].mean(), 4),
        round(scores['test_ACC'].std(), 2),
        round(scores['test_recall'].mean(), 4),
        round(scores['test_recall'].std(), 2),
        round(scores['test_f1'].mean(), 4),
        round(scores['test_f1'].std(), 2),
        round(scores['test_roc_auc'].mean(), 4),
        round(scores['test_roc_auc'].std(), 2),
        round(scores['test_ACC_B'].mean(), 4),
        round(scores['test_ACC_B'].std(), 2),
        round(scores['test_kappa'].mean(), 4),
        round(scores['test_kappa'].std(), 2),
        round(scores['test_gmean'].mean(), 4),
        round(socres['test_gmean'].std(),2)
    ]

    return results


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
    matriz = (pd.crosstab(test_labels, preds, rownames=[
              "REAL"], colnames=["PREDITO"], margins=True))
    print("Classificador: %s" % (classifier))
    print("Predições %s" % (preds))
    print("Acurácia Teste: %s" % (accu))
    print("Recall: %s" % (recall))
    print("F1: %s" % (f1))
    print("AUC: %s" % (auc))
    print("%s" % (matriz))
    return


def evaluate_model_cross(trial, classifier, model, df, f_output):

    print(classifier)
    X = df[df.columns[1:(len(df.columns) - 1)]]
    print(type(X), X.shape)

    y = df.iloc[:, -1]
    print(type(y), y.shape)

    pipe = Pipeline(steps=[
        ('MinMaxScaler', MinMaxScaler(feature_range=(0, 1))),
        ('clf', model)])

    scoring = {'ACC': 'accuracy',
               'recall': 'recall',
               'f1': 'f1',
               'roc_auc': 'roc_auc',
               'ACC_B': 'balanced_accuracy',
               'kappa': make_scorer(cohen_kappa_score),
               'gmean': make_scorer(geometric_mean_score)
               }

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_validate(pipe, X, y, cv=kfold, scoring=scoring)

    save_measures(calculate_results(trial, classifier, scores), f_output)


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
    parser.add_argument(
        '-i', '--input', help='csv format file, E.g., dataset.csv')
    parser.add_argument(
        '-o', '--output', help='CSV format file, E.g., test.csv')
    # parser.add_argument('-k', '--kmer', help='Range of k-mer, E.g., 1-mer (1) or 2-mer (1, 2) ...')
    # parser.add_argument('-e', '--entropy', help='Type of Entropy, E.g., Shannon or Tsallis')
    # parser.add_argument('-q', '--parameter', help='Tsallis - q parameter')
    args = parser.parse_args()
    finput = str(args.input)
    foutput = str(args.output)

    experiments = {
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(criterion='gini', max_depth=2, max_leaf_nodes=None, random_state=63),
        # "RandomForest": RandomForestClassifier(random_state=63, n_estimators=100),
        # "LogisticRegression": LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5),
        # "SVM": svm.SVC(gamma='scale', kernel='poly', degree=5, coef0=0.1, random_state=63),
        # "Bagging": BaggingClassifier(random_state=63),
        # "KNN": KNeighborsClassifier(),
        # "MLP": MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 2), learning_rate_init=0.001, random_state=63),
        # "Adaboost": AdaBoostClassifier(random_state=63),
        # "Catboost": CatBoostClassifier(iterations=100, random_seed=63, logging_level='Silent'),
        # "GradientBoosting": GradientBoostingClassifier(n_estimators=400, learning_rate=3.0, max_depth=1, random_state=63),
        # # "HistGradientBoosting": HistGradientBoostingClassifier(random_state=63)
    }

    classifiers = [
        "GaussianNB",
        "DecisionTree"#,
        # "RandomForest",
        # "LogisticRegression",
        # "SVM",
        # "Bagging",
        # "KNN",
        # "MLP",
        # "Adaboost",
        # "Catboost",
        # "GradientBoosting",
        # "HistGradientBoosting"
    ]

    print("Input:  ", finput)
    print("Output:  ", foutput)

    trial = 6

    for classifier in classifiers:
        model = experiments.pop(classifier)
        frame = read_dataset(finput)
        evaluate_model_cross(trial, classifier, model, frame, foutput)
        # evaluate_model_holdout(classifier, model, finput)
        del frame
        del model
