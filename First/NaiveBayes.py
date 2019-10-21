import scipy.stats
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors.kde import KernelDensity

class NaiveBayes:
    """
        Naive Bayes class with Kernel Density Estimator.

        Methods
        ----------
        - __init__ : bandwidth
            
            Init function. It creates the empty X and y train set and set the bandwidth for the KDE.
        
        - fit : X_train, y_train
        
            This function set the X and y train set woth the set given as argument.
        
        - class_prob : y_class
        
            Calculates the probability of the class y_class.
        
        - predict_proba : X
        
            Calculate the logarithm of the likelyhood of belonging to each of the classes.
        
        - predict: X
        
            Return the predicted class for each x element in X.
        
        - classifierScore: y_pred, y_real
        
            Return the accuracy score of the classifier given the predicted and the real y.
        
        - score: x, y
        
            \??

        Notes
        -----
        Naive Bayes is a Lazy Learning algorithm, so usually the model creation computation is done
        when is requested a predict operation. In this class the computation is done in the fit function.
        We still have to decide where is better to add this functionality.
    """

    def __init__(self,bandwidth):
        """
            Init function. It creates the empty X and y train set and set the bandwidth for the KDE.

            Parameters
            ----------
            - bandwidth:
                The bandwidth of the kernel is a free parameter which exhibits a strong influence on the resulting estimate. 
        """
        self.bandwidth = bandwidth
        self.X_train = np.array([])
        self.y_train = np.array([])

    def fit(self, X_train, y_train):
        """
            Fit function. This function set the X and y train set woth the set given as argument.

            Parameters
            ----------
            - X_train:
                The train set without class information.
            - y_train:
                The classes (labels) of the train set.

            Notes
            -----
            Naive Bayes is a Lazy Learning algorithm, so usually the model creation computation is done
            when is requested a predict operation. In this class the computation is done in the fit function.
            We still have to decide where is better to add this functionality.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
        self.all_class = np.unique(self.y_train)
        
        X_train_by_class = [X_train[y_train == yi] for yi in self.all_class]

        self.models_ = [KernelDensity(bandwidth=self.bandwidth).fit(Xi) for Xi in X_train_by_class]

        self.logpriors_ = [np.log(Xi.shape[0] / X_train.shape[0]) for Xi in X_train_by_class]
        
        return self

    def class_prob(self, y_class):
        """
            Class Probability function. Calculates the probability of the class y_class.

            Parameters
            ----------
            - y_class:
                The class which we want to calculate the probability (prior probability).
        """
        n = len([item for item in self.y_train if item == y_class])
        d = len(self.y_train)
        return n * 1.0 /d

    def predict(self, X):
        """
            Predict function. Returns the predicted class for each element in X.

            Parameters
            ----------
            - X:
                The validation/test set in order to predict the class for every element in X.
        """
        return self.all_class[np.argmax(self.predict_proba(X), 1)]
    
    def predict_proba(self, X):
        """
            Predict Probability function. Calculate the logarithm of the likelyhood of belonging to each of the classes.

            Parameters
            ----------
            - X:
                The validation/test set in order to calculate the logarithm of the likelyhood for each class.
        """
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = logprobs + self.logpriors_
        return result
 
    def classifierScore(self,y_pred, y_real):    
        """
            Classifier Score function. Returns the accuracy score of the classifier given the predicted and the real y.

            Parameters
            ----------
            - y_pred:
                The predicted class for every element of the validation/test set.

            - y_real:
                The real class for every element of the validation/test set.
        """
        return accuracy_score(y_real, y_pred)

    def score(self,x,y):
        """
            Score function. 

            Parameters
            ----------
            - x:
                \??
            - y:
                \??
        """
        return self.classifierScore(self.predict(x), y)
    