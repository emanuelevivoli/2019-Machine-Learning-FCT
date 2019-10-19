import scipy.stats
import numpy as np


class NaiveBayes:
    def __init__(self):
        self.X_train = np.array([])
        self.y_train = np.array([])

    def fit(self, X_train, y_train):
        print('in fit')
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.all_class = np.unique(self.y_train)

    def class_prob(self, y_class):
        n = len([item for item in self.y_train if item == y_class])
        d = len(self.y_train)
        return n * 1.0 /d

    def _joint_log_likelihood(self, X):
        print('in _joint_log_likelihood')
        X = np.array(X)
        joint_log_likelihood = np.zeros((X.shape[0], np.size(self.all_class)))
        for i in range(np.size(self.all_class)):
            joint_log_likelihood[:,i] += np.log(self.class_prob(self.all_class[i]))

        for i, x in enumerate(X):
            for y_class in self.all_class:

                prob = 0
                for j, f in enumerate(self.X_train):
                    if self.y_train[j] != y_class: continue
                    prob += self.kernel_gaussian(x - f)

                c = np.where(self.all_class == y_class)

                joint_log_likelihood[i, c] += np.log(prob)

        return joint_log_likelihood.T

    def predict(self, X):
        print('in predict')
        joint_log_likelihood = self._joint_log_likelihood(X)
        return self.all_class[np.argmax(joint_log_likelihood, axis=0)]

    def kernel_gaussian(self, x):
        print('in kernel_gaussian')
        x = np.array(x)
        res = 1
        for col in x:
            res *= scipy.stats.norm(0, 1).pdf(col)
        return res