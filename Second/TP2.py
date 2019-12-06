from utils import *

# # libraries for Dimensionality Reduction
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.manifold import Isomap

# libraries for features selection
from feature_selection import features_creation

# ANOVA F-test
from sklearn.feature_selection import f_classif
from sklearn import datasets

# library for standardize features
from sklearn.preprocessing import StandardScaler

# libraries for epsilon parameter
from sklearn.neighbors import KNeighborsClassifier

# libraries for Clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# libraries for Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

# libraries for general utilities
# import numpy as np 
# %matplotlib widgets
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# library for parse input command specifications
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')

# parser input commands
parser.add_argument('--seed', 
                    default=42,
                    help='[default False] is the seed for the randomize processes (shuffle)')

args = parser.parse_args()

set_seed(args.seed)

# global variables from work specifications
NUMBER_FEATURES = 6
NEIGHBOURS = 5
CORRELATION_LIMIT = 0.6
ALPHA = 0.05


# main function
def main():
    ##
    ## DATASET LOADING

    # create the loader class and load dataset and labels
    loader = Loader().load()

    # get dataset, labels, and dataset splitted by classes
    dataset = loader.getDataset()
    labels = loader.getLabels()
    class_split_dataset = loader.getClassSplitDataset()

    ##
    ## FEATURES CREATION

    # features creation of 18 features
    features = features_creation(dataset, NUMBER_FEATURES)

    ##
    ## CHECK FEATURES IMPORTANCE WITH SVM
    importance_test_SVM(features, labels)

    # standardization
    features_std, ids, labels = features_standardization(features, labels)

    # create visualizations
    corr = features_correlation_matrix(features_std, title='correlation_matrix')
    indexes = correlated_features(corr, CORRELATION_LIMIT)
    correlated_scatter_matrix(features_std, indexes, title='correlated_scatter_matrix')

    ##
    ## FEATURES REMOVE with correlation matrix

    features_sel = remove_features(features_std, indexes)

    ##
    ## FEATURES REMOVE with anova f test

    features_sel = anova_f_test_selection(features_sel, labels, alpha=ALPHA)


    ##
    ##  CLUSTERING DBSCAN
    ##

    dbscan_by_num, eps_matrix = DBSCAN_clustering(features, title='find_eps_params')

    score_by_num = calculate_scores(labels, start_feat, final_feat, dbscan_by_num)

    win_ids = calculate_bests_indexes(score_by_num)

    win_dic = indexes_to_dict(win_ids)

    save_3d_plot(win_dic, eps_matrix, dbscan_by_num, title='cluster')

    ##
    ##  CLUSTERING KMEANS
    ##








if __name__ == "__main__":
    print("start main")
    main()
    print("end main")