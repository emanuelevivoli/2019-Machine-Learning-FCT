import numpy as np
from sklearn import svm
import NaiveBayes as nb
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

N_SPLIT = 5
SEED = 42

class Dataset:

    def __init__(self, array):
        self.data = array[0]
        self.labels = array[1]

class Loader:

    def __init__(self):

        self.train = []
        self.test = []
        
        self.mean = []
        self.std = []
    
    def set_train(self, train_file_name):
    
        self.train = Dataset(self.load(True, train_file_name))

        return self

    def set_test(self, test_file_name):

        self.test = Dataset(self.load(False, test_file_name))

        return self

    def load(self, boolean, file_name):

        # take data from the file and put all in a ndArray
        mat = np.loadtxt(file_name, delimiter='\t')
        # randomize the rows
        # np.random.seed(SEED)
        np.random.shuffle(mat)

        # Ys take only the labels column
        # Xs take the data without labels
        Ys = mat[:, [-1]]
        Xs = mat[:, :-1]

        # standardize
        Xs = self.standardize(boolean, Xs)

        return [Xs, Ys]

    def standardize(self, boolean, Xs):

        if boolean:
            # mean and std calculus
            self.means = np.mean(Xs, 0)
            self.std = np.std(Xs, 0)
        
        # standardize Xs values
        Xs = (Xs - self.means) / self.std

        return Xs

    def get_train(self):

        return self.train
    
    def get_test(self):

        return self.test

class Model:

    def __init__(self, name="Interface"):
        self.name = name
        self.model = None

    def print_name(self):
        print(self.name)

    def build_model(self, gamma):
        print("** Error, build_model(self, gamma) not overridden **")

    def get_result(self, X_train, y_train, X_val, y_val, gamma):
        self.build_model(gamma)
        # self.print_name()
        self.model.fit(X_train, y_train.ravel())

        t_predictions = self.model.predict(X_train)
        v_predictions = self.model.predict(X_val)
        
        ern, eri = find_error_values(t_predictions, y_train)
        el_num = np.size(t_predictions)
        t_accuracy = (1 - ern / el_num)

        ern, eri = find_error_values(v_predictions, y_val)
        el_num = np.size(v_predictions)
        v_accuracy = (1 - ern / el_num)

        return v_accuracy, t_accuracy, ern, eri

class SVM(Model):

    def __init__(self, name="SVM"):
        super().__init__(name)
        self.interval = np.arange(start=0.2 , stop=6.2 , step=0.2)
        self.param_name = "gamma"

    def build_model(self, gamma="auto"):
        self.model = svm.SVC(C=1, gamma=gamma)
    
    def get_result(self, X_train, y_train, X_val, y_val, gamma):
        return super().get_result(X_train, y_train, X_val, y_val, gamma)

    def get_interval(self):
        return self.interval

class NB(Model):

    def __init__(self, name="NB"):
        super().__init__(name)
        self.interval = np.arange(start=0.02, stop=0.62, step=0.02)
        self.param_name = "bandwidth"

    def build_model(self, gamma):
        self.model = nb.NaiveBayes(gamma)
    
    def get_result(self, X_train, y_train, X_val, y_val, gamma):
        return super().get_result(X_train, y_train, X_val, y_val, gamma)

    def get_interval(self):
        return self.interval

class GaussNB(Model):
    
    def __init__(self, name="GaussNB"):
        super().__init__(name)

    def build_model(self, gamma):
        self.model = GaussianNB()
    
    def get_result(self, X_train, y_train, X_val, y_val, gamma=None):
        return super().get_result(X_train, y_train, X_val, y_val, gamma)

class Result():

    def __init__(self, model, best_gamma, valid_results, train_results):
        self.model = model
        self.best_gamma = best_gamma
        self.valid_results = valid_results
        self.train_results = train_results

    def save_plot(self):
        plt.figure()
        t = self.model.get_interval()
        plt.plot(t, self.train_results, 'bo', label='training error')
        plt.plot(t, self.valid_results, 'ro', label='cross-validation error')
        plt.xlabel('error')
        plt.ylabel(f'{self.model.param_name}')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(f'{self.model.name}.png', bbox_inches='tight')
        
        # plt.show()
        # plt.close()




def find_error_values(a, b):
    error_n = 0
    error_indexes = []
    for i in range(0, np.size(a, 0)):
        if (a[i] != b[i]):
            error_n = error_n + 1
            error_indexes.append(i)
    return error_n, error_indexes

def gamma_optimizing(model, X, y):

    valid_results = []
    train_results = []

    skf = StratifiedKFold(n_splits=N_SPLIT)

    min_valid_error = float('inf')
    min_train_error = float('inf')

    best_gamma = 0

    for g in model.get_interval():

        sum_valid_acc = 0
        sum_train_acc = 0

        for train_index, test_index in skf.split(X, y):
            
            X_train, y_train = X[train_index], y[train_index] 
            X_test , y_test  = X[test_index] , y[test_index]

            one_fold_valid_acc, one_fold_train_acc, _, _ = model.get_result(X_train, y_train, X_test, y_test, gamma=g)
            
            sum_valid_acc += one_fold_valid_acc
            sum_train_acc += one_fold_train_acc

        k_fold_valid_err = (1 - sum_valid_acc / N_SPLIT)
        k_fold_train_err = (1 - sum_train_acc / N_SPLIT)

        valid_results.append(k_fold_valid_err)
        train_results.append(k_fold_train_err)

        if k_fold_valid_err < min_valid_error:
            min_valid_error = k_fold_valid_err
            best_gamma = g

    return Result(model, best_gamma, valid_results, train_results)




def normal_comparing(dic, N):

    best_classifier    = None
    minimum_err_number = N

    inf_extreams = []
    for name, values in dic.items():
        p0 = 1 - values[0]

        err_number = values[1]
        sigma = np.sqrt( N * p0 * (1 - p0))
        inf_extreams.append([1.96 * sigma])

        print(name, " = ", err_number, "\u00B1", 1.96 * sigma)

    better_performance = min(inf_extreams)
    indexes = [i for i, x in enumerate(inf_extreams) if x == better_performance]

    print("\nAccording to the approximate normal test:\n", ' and '.join([list(dic.keys()) [i] for i in indexes]),
          " classifier(s) seems to have better performance")

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

