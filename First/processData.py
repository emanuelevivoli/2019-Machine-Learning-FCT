# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 01:29:34 2019

@author: simon
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import NaiveBayes as nb

from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    # prima parte: da completare -----------------------------------
    print("\n--------------------first part-------------------------\n")

    data = load_data('TP1_train.tsv')
    x = divideTrainingAndValidation(data[0], data[1], 0.7)
    y, a, b = gaussianNbResult(x[0], x[1], x[2], x[3])
    z, a, b = supportVectorMachineResult(x[0], x[1], x[2], x[3])
    my, a, b = myNbResults(x[0], x[1], x[2], x[3], chooseBestBandWidth(x[0], x[1]))
    # my=kernelDensityResults(x[0],x[1])
    print([y, z, my])

    # tuning-------------------------------------------------------------
    print("\n--------------------tuning stuff-------------------------\n")
    cl_name = "svm"
    plot_name = "SVM.png"
    param_option = {
        "p_name": "gamma",
        "p_start": 0.2,
        "p_end": 6,
        "p_step": 0.2
    }
    best_gamma = classifier_parameter_tuning(x[0], x[1], x[2], x[3], cl_name, param_option, plot_name)

    cl_name = "myNB"
    plot_name = "NB.png"
    param_option = {
        "p_name": "bandwidth",
        "p_start": 0.02,
        "p_end": 0.6,
        "p_step": 0.02
    }
    best_bw = classifier_parameter_tuning(x[0], x[1], x[2], x[3], cl_name, param_option, plot_name)

    # test finale per comparare i classificatori ------------------
    print("\n--------------------comparing classifier-------------------------\n")

    # carico i dati del test set ma normalizzandoli come nel train set
    real_x = load_data_mean_stdevs('TP1_test.tsv', data[2], data[3])

    # per ogni classificatore
    # accuracy, numero errori, indici errati
    gnb_a, gnb_n, gnb_i = gaussianNbResult(data[0], data[1], real_x[0], real_x[1])
    # svm_a, svm_n, svm_i = supportVectorMachineResult(data[0],data[1],real_x[0],real_x[1],best_gamma)
    svm_a, svm_n, svm_i = supportVectorMachineResult(data[0], data[1], real_x[0], real_x[1])
    # my_a, my_n, my_i = myNbResults(data[0],data[1],real_x[0],real_x[1],best_bw)
    my_a, my_n, my_i = myNbResults(data[0], data[1], real_x[0], real_x[1], 0.16)

    results = {
        "gNB": [gnb_a, gnb_n],
        "svm": [svm_a, svm_n],
        "myNB": [my_a, my_n]
    }

    compare_using_normal_test(results, real_x[0])

    compare_using_mcnemar("gNB", gnb_i, "svm", svm_i)
    compare_using_mcnemar("myNB", my_i, "gNB", gnb_i)
    compare_using_mcnemar("svm", svm_i, "myNB", my_i)

    print()
    return [gnb_a, svm_a, my_a]


def load_data(file_name):
    # prende i dati da file, li mette in un ndArray
    mat = np.loadtxt(file_name, delimiter='\t')
    # randomizzo le righe
    np.random.shuffle(mat)
    # prendo l'ultima colonna (quella delle classe)
    Ys = mat[:, [-1]]
    # prendo le righe, eccetto l'ultima colonna
    Xs = mat[:, :-1]
    # calcola la media per ogni colonna, restituisce l'array con le medie
    means = np.mean(Xs, 0)
    # calcola la standard deviation per ogni colonna, restituisce l'array con le standard deviation
    stdevs = np.std(Xs, 0)
    # valori standardizzati
    Xs = (Xs - means) / stdevs
    # restituisce una tupla
    # Xs son le righe standardizzate
    # Ys le relative classi
    # means è la media per ogni attributo
    # stdevs è la standard deviation per ogni attributo
    return (Xs, Ys, means, stdevs)


def load_data_mean_stdevs(file_name, means, stdevs):
    # prende i dati da file, li mette in un ndArray
    mat = np.loadtxt(file_name, delimiter='\t')
    # randomizzo le righe
    np.random.shuffle(mat)
    # prendo l'ultima colonna (quella delle classe)
    Ys = mat[:, [-1]]
    # prendo le righe, eccetto l'ultima colonna
    Xs = mat[:, :-1]
    # valori standardizzati
    Xs = (Xs - means) / stdevs
    # restituisce una tupla
    # Xs son le righe standardizzate
    # Ys le relative classi
    # means è la media per ogni attributo
    # stdevs è la standard deviation per ogni attributo
    return (Xs, Ys)


def divideTrainingAndValidation(Xs, Ys, percentage):
    len = Ys.size
    trainingLen = int(len * percentage)
    trainingXs = Xs[0:trainingLen]
    trainingYs = Ys[0:trainingLen]
    validationXs = Xs[trainingLen:]
    validationYs = Ys[trainingLen:]
    return (trainingXs, trainingYs, validationXs, validationYs)


def find_error_values(a, b):
    error_n = 0
    error_indexes = []
    for i in range(0, np.size(a, 0)):
        if (a[i] != b[i]):
            error_n = error_n + 1
            error_indexes.append(i)
    return error_n, error_indexes


def gaussianNbResult(Xs, Ys, Xvalidation, Yvalidation):
    clf = GaussianNB()
    clf.fit(Xs, Ys.ravel())
    # print("printng ",clf.predict(Xvalidation))
    predictions = clf.predict(Xvalidation)
    ern, eri = find_error_values(predictions, Yvalidation)
    el_num = np.size(predictions)
    accuracy = (1 - ern / el_num)
    # print("ern",ern,"eri",eri)
    return accuracy, ern, eri


# di questo dobbiamo scegliere il valore migliore di gamma
# con una cross-validation sul training set
def supportVectorMachineResult(Xs, Ys, Xvalidation, Yvalidation):
    clf = svm.SVC(C=1, gamma='auto')
    clf.fit(Xs, Ys.ravel())
    # print("printng ",clf.predict(Xvalidation))
    predictions = clf.predict(Xvalidation)
    ern, eri = find_error_values(predictions, Yvalidation)
    el_num = np.size(predictions)
    accuracy = (1 - ern / el_num)
    # print("ern",ern,"eri",eri)
    return accuracy, ern, eri


################################################################


def chooseBestBandWidth(data, y):
    params = {'bandwidth': np.arange(0.02, 0.6, 0.02)}
    cv = StratifiedShuffleSplit(test_size=0.2)
    grid = GridSearchCV(KernelDensity(), params, cv=cv, iid=False)
    grid.fit(data, y)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    return grid.best_estimator_.bandwidth


# prova diverse bandwith e ritorna il migliore estimatore
def chooseBestEstimator(data, y):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.arange(0.02, 0.6, 0.02)}
    cv = StratifiedShuffleSplit(test_size=0.2)
    grid = GridSearchCV(KernelDensity(), params, cv=cv, iid=False)
    grid.fit(data, y)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    return kde


def kernelDensityResults(Xs, Ys):
    kde = chooseBestEstimator(Xs, Ys).fit(Xs, Ys)
    return np.exp(kde.score_samples(Xs))


def myNbResults(Xs, Ys, Xvalidation, Yvalidation, es):
    clf = nb.NaiveBayes(es)
    clf.fit(Xs, Ys.ravel())
    # print("printng ",ourNB.predict(Xvalidation))
    predictions = clf.predict(Xvalidation)
    ern, eri = find_error_values(predictions, Yvalidation)
    el_num = np.size(predictions)
    accuracy = (1 - ern / el_num)
    # print("ern",ern,"eri",eri)
    return accuracy, ern, eri


def compare_using_normal_test(dic, a):
    print()
    test_set_size = np.size(a, 0)
    # print("test_set_size",test_set_size)

    name_best_classifier = ""
    minimum = test_set_size

    for name, values in dic.items():
        # print(values[0])
        error = 1 - values[0]
        number_of_errors = values[1]
        temp = number_of_errors - (error * 1.96)
        if (temp < minimum):
            minimum = temp
            name_best_classifier = name
        print(name, " = ", number_of_errors, "\u00B1", (error * 1.96))

    print("\nAccording to the approximate normal test:\n", name_best_classifier,
          " classifier seems to have better performance")

    return


def first_present_second_not(indexesA, indexesB):
    result = 0
    for el in indexesA:
        if el not in indexesB:
            result = result + 1
    return result


def compare_using_mcnemar(nameFirst, indexesFirst, nameSecond, indexesSecond):
    print()

    valueFirst = first_present_second_not(indexesFirst, indexesSecond)
    valueSecond = first_present_second_not(indexesSecond, indexesFirst)

    numerator = (abs(valueFirst - valueSecond) - 1) ** 2
    sub = valueFirst + valueSecond
    if (sub == 0):
        print("Impossible to compare due to the data")
    else:
        result = numerator / sub
        print(nameFirst, " vs. ", nameSecond, " = ", result)
        print("According to the McNemar test between", nameFirst, " and ", nameSecond, ": ")
        if (result > 3.84):
            print(" the performance is different")
        else:
            print(" the difference of performance is not significant")


def classifier_parameter_tuning(x, y, xTrain, yTrain, cl_name, param_option, plot_name):
    print()
    print("cl_name ", cl_name)
    print("param_option ", param_option["p_name"])
    print("param_option ", param_option["p_start"])
    print("param_option ", param_option["p_end"])
    print("param_option ", param_option["p_step"])
    print("plot_name ", plot_name)
    print()

    # dovrebbe tornare il parametro migliore
    # e stampare il plot con training e cross validation error
    return ""



