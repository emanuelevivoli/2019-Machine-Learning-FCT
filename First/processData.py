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
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

# SEED = 42

def main():
    data = load_data('TP1_train.tsv')
    # x = divideTrainingAndValidation(data[0],data[1],0.7)

    print("\n--------------------training error-------------------------\n")

    svm_te = svm_te_gamma(data[0], data[1])
    mynb_te = mynb_te_bandwidth(data[0], data[1])

    # print(svm_te)
    # print("\n",mynb_te)

    print("\n--------------------cross-validation error-------------------------\n")

    best_gamma, svm_cve = svm_cve_gamma(data[0], data[1])
    best_bw, mynb_cve = mynb_cve_bandwidth(data[0], data[1])

    # print("\n",best_gamma," - ",svm_cve)
    # print("\n",best_bw," - ",mynb_cve)

    plot_together_svm(svm_te, svm_cve)
    plot_together_mynb(mynb_te, mynb_cve)

    print("\n--------------------comparing classifier-------------------------\n")

    # carico i dati del test set ma normalizzandoli come nel train set
    real_x = load_data_mean_stdevs('TP1_test.tsv', data[2], data[3])

    # per ogni classificatore
    # accuracy, numero errori, indici errati
    gnb_a, gnb_n, gnb_i = gaussianNbResult(data[0], data[1], real_x[0], real_x[1])
    # svm_a, svm_n, svm_i = supportVectorMachineResult(data[0],data[1],real_x[0],real_x[1],best_gamma)
    svm_a, svm_n, svm_i = supportVectorMachineResult(data[0], data[1], real_x[0], real_x[1], best_gamma)
    # my_a, my_n, my_i = myNbResults(data[0],data[1],real_x[0],real_x[1],best_bw)
    my_a, my_n, my_i = myNbResults(data[0], data[1], real_x[0], real_x[1], best_bw)

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


# ------functions to load the data from the file------

def load_data(file_name):
    # prende i dati da file, li mette in un ndArray
    mat = np.loadtxt(file_name, delimiter='\t')
    # randomizzo le righe
    
    # np.random.seed(SEED)
    # np.random.shuffle(mat)
    
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
    
    # np.random.seed(SEED)
    # np.random.shuffle(mat)
    
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


# ------function divide a set in training and validation set------

def divideTrainingAndValidation(Xs, Ys, percentage):
    len = Ys.size
    trainingLen = int(len * percentage)
    trainingXs = Xs[0:trainingLen]
    trainingYs = Ys[0:trainingLen]
    validationXs = Xs[trainingLen:]
    validationYs = Ys[trainingLen:]
    return (trainingXs, trainingYs, validationXs, validationYs)


# ------functions to get the results from classifiers------

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


def supportVectorMachineResult(Xs, Ys, Xvalidation, Yvalidation, gamma='auto'):
    clf = svm.SVC(C=1, gamma=gamma)
    clf.fit(Xs, Ys.ravel())
    # print("printng ",clf.predict(Xvalidation))
    predictions = clf.predict(Xvalidation)
    ern, eri = find_error_values(predictions, Yvalidation)
    el_num = np.size(predictions)
    accuracy = (1 - ern / el_num)
    # print("ern",ern,"eri",eri)
    return accuracy, ern, eri


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


# ------functions to calculate the training error------

def svm_te_gamma(x, y, start=0.2, end=6.2, step=0.2):
    results = []
    for g in np.arange(start, end, step):
        a, en, er = supportVectorMachineResult(x, y, x, y, g)
        results.append(1 - a)
    # print("svm_te_gamma",results)
    return results


def mynb_te_bandwidth(x, y, start=0.02, end=0.62, step=0.02):
    results = []
    for bw in np.arange(start, end, step):
        a, en, er = myNbResults(x, y, x, y, bw)
        results.append(1 - a)
    return results


# ------functions for the cross-validation------

def svm_cve_gamma(x, y, start=0.2, end=6.2, step=0.2):
    results = []
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(x, y)
    # print(skf)
    min_error = -1
    best_gamma = 0
    for g in np.arange(start, end, step):
        tempValue = 0
        cycle = 0
        for train_index, test_index in skf.split(x, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            a, en, er, = supportVectorMachineResult(X_train, y_train, X_test, y_test, gamma=g)
            tempValue += a
            # print(tempValue)
            cycle += 1
        # print()
        g_mean = (1 - tempValue / cycle)
        results.append(g_mean)
        if min_error == -1 or g_mean < min_error:
            min_error = g_mean
            best_gamma = g

    return best_gamma, results


def mynb_cve_bandwidth(x, y, start=0.02, end=0.62, step=0.02):
    results = []
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(x, y)
    # print(skf)
    min_error = -1
    best_bw = 0
    for bw in np.arange(start, end, step):
        tempValue = 0
        cycle = 0
        for train_index, test_index in skf.split(x, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            a, en, er, = myNbResults(X_train, y_train, X_test, y_test, bw)
            tempValue += a
            # print(tempValue)
            cycle += 1
        # print()
        bw_mean = (1 - tempValue / cycle)
        results.append(bw_mean)
        if min_error == -1 or bw_mean < min_error:
            min_error = bw_mean
            best_bw = bw

    return best_bw, results


# ------functions to compare classification model------

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


# ------functions to plot training and validation error------

def plot_together_svm(svm_te, svm_cve):
    plt.figure()
    print("\na ", svm_te)
    print("\nb ", svm_cve)
    t = np.arange(0.2, 6.2, 0.2)
    plt.plot(t, svm_te, 'bo', label='training error')
    plt.plot(t, svm_cve, 'ro', label='cross-validation error')
    plt.xlabel('error')
    plt.ylabel('gamma')
    plt.grid(True)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.savefig('SVM_old.png', bbox_inches='tight')
    plt.show()

    plt.close()


def plot_together_mynb(te, cve):
    plt.figure()
    print("\na ", te)
    print("\nb ", cve)
    t = np.arange(0.02, 0.62, 0.02)
    plt.plot(t, te, 'bo', label='training error')
    plt.plot(t, cve, 'ro', label='cross-validation error')
    plt.xlabel('error')
    plt.ylabel('bandwidth')
    plt.grid(True)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.savefig('NB_old.png', bbox_inches='tight')
    plt.show()
    plt.close()

main()
