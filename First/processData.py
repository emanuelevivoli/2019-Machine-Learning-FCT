# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 01:29:34 2019

@author: simon
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score as score

def load_data(file_name):
     #prende i dati da file, li mette in un ndArray
     mat = np.loadtxt(file_name,delimiter='\t')
     #randomizzo le righe
     np.random.shuffle(mat)
     #prendo l'ultima colonna (quella delle classe)
     Ys = mat[:,[-1]]
     #prendo le righe, eccetto l'ultima colonna
     Xs = mat[:,:-1]
     #calcola la media per ogni colonna, restituisce l'array con le medie
     means = np.mean(Xs,0)
     #calcola la standard deviation per ogni colonna, restituisce l'array con le standard deviation
     stdevs = np.std(Xs,0)
     #valori standardizzati
     Xs = (Xs-means)/stdevs
     #restituisce una tupla
     #Xs son le righe standardizzate
     #Ys le relative classi
     #means è la media per ogni attributo
     #stdevs è la standard deviation per ogni attributo
     return (Xs,Ys,means,stdevs)

def divideTrainingAndValidation(Xs,Ys,percentage):
    len = Ys.size
    trainingLen = int(len*percentage)
    trainingXs = Xs[0:trainingLen]
    trainingYs = Ys[0:trainingLen]
    validationXs = Xs[trainingLen:]
    validationYs = Ys[trainingLen:]
    return(trainingXs, trainingYs, validationXs, validationYs)
    
def gaussianNbResult(Xs,Ys,Xvalidation,Yvalidation):
    clf = GaussianNB()
    clf.fit(Xs, Ys.ravel())
    return clf.score(Xvalidation,Yvalidation)

#di questo dobbiamo scegliere il valore migliore di gamma
#con una cross-validation sul training set    
def supportVectorMachineResult(Xs,Ys,Xvalidation,Yvalidation):
    clf = svm.SVC(C=1,gamma='auto')
    clf.fit(Xs, Ys.ravel())
    return clf.score(Xvalidation,Yvalidation)
    




