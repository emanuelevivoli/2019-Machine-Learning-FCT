import scipy.stats
import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.neighbors.kde import KernelDensity

class NaiveBayes:
    def __init__(self,bw):
        self.bandwidth = bw
        self.X_train = np.array([])
        self.y_train = np.array([])

    def fit(self, X_train, y_train):
        print('in fit')
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
        self.all_class = np.unique(self.y_train)
        
        
        training_sets = [X_train[y_train == yi] for yi in self.all_class]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X_train.shape[0])
                           for Xi in training_sets]
        
        return self

    def class_prob(self, y_class):
        n = len([item for item in self.y_train if item == y_class])
        d = len(self.y_train)
        return n * 1.0 /d

    def predict(self, X):
        return self.all_class[np.argmax(self.predict_proba(X), 1)]
    
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = logprobs + self.logpriors_
        return result

    # x l'array con le classi predette dal nostro classificatore
    # y l'array con le classi effettive   
    def classifierScore(self,x,y):    
        return accuracy_score(y, x)

    def score(self,x,y):
        return self.classifierScore(self.predict(x), y)
    
    
        
    