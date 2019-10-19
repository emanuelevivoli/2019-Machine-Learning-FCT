# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 01:29:34 2019

@author: simon
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
import NaiveBayes as nb

from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV

def main():
    data = load_data('TP1_train.tsv')
    x = divideTrainingAndValidation(data[0],data[1],0.7)
    y = gaussianNbResult(x[0],x[1],x[2],x[3])
    z = supportVectorMachineResult(x[0],x[1],x[2],x[3])
    
    my = myNbResults(x[0],x[1],x[2],x[3],chooseBestBandWidth(x[0],x[1]))
    #my=kernelDensityResults(x[0],x[1])
    
    return (y,z,my)

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
    
################################################################


def chooseBestBandWidth(data,y):
    params = {'bandwidth': np.arange(0.02, 0.6, 0.02)}
    grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
    grid.fit(data,y)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    return grid.best_estimator_.bandwidth

#prova diverse bandwith e ritorna il migliore estimatore
def chooseBestEstimator(data,y):
    # use grid search cross-validation to optimize the bandwidth
    params = {'bandwidth': np.arange(0.02, 0.6, 0.02)}
    grid = GridSearchCV(KernelDensity(), params, cv=5, iid=False)
    grid.fit(data,y)
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    kde = grid.best_estimator_
    return kde

def kernelDensityResults(Xs,Ys):
    kde = chooseBestEstimator(Xs,Ys).fit(Xs,Ys)
    return np.exp(kde.score_samples(Xs))


def myNbResults(Xs,Ys,Xvalidation,Yvalidation,es):
    ourNB = nb.NaiveBayes(es)
    ourNB.fit(Xs, Ys.ravel())
    return ourNB.score(Xvalidation,Yvalidation.ravel())
    
    
