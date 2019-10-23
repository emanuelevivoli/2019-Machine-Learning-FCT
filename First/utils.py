import numpy as np
from sklearn import svm
import NaiveBayes as nb
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from itertools import combinations
import csv

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
        np.random.seed(SEED)
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

    def build_model(self, params):
        print("** Error, build_model(self, params) not overridden **")

    def get_result(self, X_train, y_train, X_val, y_val, params):
        self.build_model(params)
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

    def predict(self, X_test, y_test):
        test_predictions = self.model.predict(X_test)

        ern, eri = find_error_values(test_predictions, y_test)
        el_num = np.size(test_predictions)
        test_accuracy = (1 - ern / el_num)

        return test_accuracy


class SVM(Model):

    def __init__(self, name="SVM"):
        super().__init__(name)
        self.interval = np.arange(start=0.2 , stop=6.2 , step=0.2)
        self.param_name = "gamma"

    def build_model(self, params=["auto", 1]):
        self.model = svm.SVC(C=params[1], gamma=params[0])
    
    def get_result(self, X_train, y_train, X_val, y_val, params):
        return super().get_result(X_train, y_train, X_val, y_val, params)

    def get_interval(self):
        return self.interval

class NB(Model):

    def __init__(self, name="NB"):
        super().__init__(name)
        self.interval = np.arange(start=0.02, stop=0.62, step=0.02)
        self.param_name = "bandwidth"

    def build_model(self, params):
        self.model = nb.NaiveBayes(params[0])
    
    def get_result(self, X_train, y_train, X_val, y_val, params):
        return super().get_result(X_train, y_train, X_val, y_val, params)

    def get_interval(self):
        return self.interval

class GaussNB(Model):
    
    def __init__(self, name="GaussNB"):
        super().__init__(name)

    def build_model(self, params):
        self.model = GaussianNB()
    
    def get_result(self, X_train, y_train, X_val, y_val, params=None):
        return super().get_result(X_train, y_train, X_val, y_val, params)

class Result():

    def __init__(self, model, best_param, valid_results, train_results):
        self.name = model.name
        self.param_name = model.param_name
        self.interval = model.get_interval()

        self.best_param = best_param
        self.valid_results = valid_results
        self.train_results = train_results

    def save_plot(self):
        plt.figure()
        t = self.interval
        plt.plot(t, self.train_results, 'bo', label='training error')
        plt.plot(t, self.valid_results, 'ro', label='cross-validation error')
        plt.ylabel('error')
        plt.xlabel(f'{self.param_name}')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.savefig(f'{self.name}.png', bbox_inches='tight')
        plt.close()


def find_error_values(a, b):
    error_n = 0
    error_indexes = []
    for i in range(0, np.size(a, 0)):
        if (a[i] != b[i]):
            error_n = error_n + 1
            error_indexes.append(i)
    return error_n, error_indexes

def param_optimizing(model, X, y):

    valid_results = []
    train_results = []

    skf = StratifiedKFold(n_splits=N_SPLIT)

    min_valid_error = float('inf')
    min_train_error = float('inf')

    best_param = 0

    for p in model.get_interval():

        sum_valid_acc = 0
        sum_train_acc = 0

        for train_index, test_index in skf.split(X, y):
            
            X_train, y_train = X[train_index], y[train_index] 
            X_test , y_test  = X[test_index] , y[test_index]

            one_fold_valid_acc, one_fold_train_acc, _, _ = model.get_result(X_train, y_train, X_test, y_test, params=[p, 1])
            
            sum_valid_acc += one_fold_valid_acc
            sum_train_acc += one_fold_train_acc

        k_fold_valid_err = (1 - sum_valid_acc / N_SPLIT)
        k_fold_train_err = (1 - sum_train_acc / N_SPLIT)

        valid_results.append(k_fold_valid_err)
        train_results.append(k_fold_train_err)

        if k_fold_valid_err < min_valid_error:
            min_valid_error = k_fold_valid_err
            best_param = p

    return Result(model, best_param, valid_results, train_results)


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

def set_substraction(set_A, set_B):
    # [A and B are set] A.difference(B) is equal to the elements present in A but not in B
    return len(set(set_A).difference(set(set_B)))

def mcnemar_comparing(dic):

    # keys_pairs = set([ (a, b) for a in dic.keys() for b in dic.keys() if a != b])

    for key_A, key_B in combinations(dic.keys(),2):
        name_A = key_A
        name_B = key_B

        err_indexes_A = dic[key_A][2]
        err_indexes_B = dic[key_B][2]

        e01 = set_substraction(err_indexes_A, err_indexes_B)
        e10 = set_substraction(err_indexes_B, err_indexes_A)

        numer = (abs(e01 - e10) - 1) ** 2
        denom = e01 + e10
        if (denom == 0):
            print(f'\n {name_A} vs. {name_B} \n Impossible to compare due to the data')
        else:
            chisqrt = numer / denom
            print(f'\n {name_A} vs. {name_B} = {chisqrt}')
            print(f'According to the McNemar test between {name_A} and {name_B} :')
            
            if (chisqrt > 3.84):
                print(' the performance is different')
            else:
                print(' the difference of performance is not significant')


def hyperparameter_optimization(X_train, y_train, X_test , y_test):
    
    import rbfopt
    # X_test , y_test
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=SEED, stratify=y_train)

    ''' Radial Basis Function hyperparameters optimization '''

    csv_evaluations_file = "evaluations.csv"

    # hyperparameters domains
    hyp_domains = {"gamma": (0.02, 0.62), "C": (1.e-03, 1.e+02)}
    num_evaluations = 25

    print("Beginning hyperparameters optimization with RBF")
    csv_header = ['hyp_opt', 'gamma', 'C', 'train_err', 'val_err', 'test_err']

    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(csv_header)
    
    hyp_opt = "RBF"
    var_lower = [ hyp_domains["gamma"][0], hyp_domains["C"][0] ]
    var_upper = [ hyp_domains["gamma"][1], hyp_domains["C"][1] ]

    def evaluate_RBF(hyperparameters):

        gamma, C = hyperparameters[0], hyperparameters[1]

        svm_model = SVM() 

        svm_val_acc, svm_train_acc, _, _ = \
                            svm_model.get_result(X_train, y_train, X_val, y_val, params=[gamma, C])

        svm_test_acc = svm_model.predict(X_test , y_test)

        with open(csv_evaluations_file, mode='a') as file_csv:
            file_csv = csv.writer(file_csv, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            # ['hyp_opt', 'gamma', 'C', 'train_accuracy', 'val_accuracy', 'test_accuracy']
            file_csv.writerow([hyp_opt, str(gamma), str(C), str(1 - svm_train_acc), str(1 - svm_val_acc), str(1 - svm_test_acc)])

        return 1 - svm_test_acc

    bb = rbfopt.RbfoptUserBlackBox(2, var_lower, var_upper, ['R', 'R'], evaluate_RBF)
    # minlp_solver_path='/Applications/bonmin', 
    settings = rbfopt.RbfoptSettings(minlp_solver_path='/Applications/bonmin', max_evaluations=num_evaluations, target_objval= 0.0)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()
    print("Results with RBF optimizer: " + str({"target": val, "gamma": x[0], "C": x[1]}) + "\n")
    with open(csv_evaluations_file, mode='a') as file_csv:
        file_csv = csv.writer(file_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        file_csv.writerow(['', '', '', '', '', '', '', ''])

    return x
