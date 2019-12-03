from utils import *

# libraries for Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap

# libraries for features selection
# ANOVA F-test
from sklearn.feature_selection import f_classif
from sklearn import datasets
# 

# libraries for epsilon parameter
from sklearn.neighbors import NearestNeighbors

# libraries for Clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

# libraries for Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

# libraries for general utilities
import numpy as np 
from matplotlib import pyplot as plt

# library for parse input command specifications
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')

# parser input commands
parser.add_argument('--seed', 
                    default=None,
                    help='[default False] is the seed for the randomize processes (shuffle)')

args = parser.parse_args()

set_seed(args.seed)

# global variables from work specifications
NUMBER_FEATURES = 6
NEIGHBOURS = 5

# main function
def main():
    # create the loader class and load dataset and labels
    loader = Loader().load()

    # get dataset, labels, and dataset splitted by classes
    dataset = loader.getDataset()
    labels = loader.getLabels()
    class_split_dataset = loader.getClassSplitDataset()

    ##
    ## FEATURES CREATION
    ##

    # extract 6 features wirh PCA
    pca = PCA(n_components=NUMBER_FEATURES)
    pca_dataset_embedded = pca.fit_transform(dataset)

    # extract 6 features wirh t-sne
    tsne = TSNE(n_components=NUMBER_FEATURES, method='exact')
    tsne_dataset_embedded = tsne.fit_transform(dataset)

    # extract 6 features wirh isomap
    isomap = Isomap(n_components=NUMBER_FEATURES)
    isomap_dataset_embedded = isomap.fit_transform(dataset)

    # features concatenate (6,6,6) = 18 features
    features = np.concatenate((pca_dataset_embedded,tsne_dataset_embedded, isomap_dataset_embedded), axis=1)

    ##
    ## FEATURES EXTRACTION
    ##

    # selection with ANOVA F-test
    f,prob = f_classif(features, labels)
    print(f)
    print(prob)


    ##
    ##  CLUSTERING
    ##

    # selecting manually the epsilon params for DBSCAN
    eps = findEpsParams(features)

    # clustering with DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=NEIGHBOURS)
    dbscan_labels = dbscan.fit_predict(dataset)

    # clustering with K-MEANS
    kmeans = KMeans(n_clusters=3)
    kmeans_labels = kmeans.fit_predict(dataset)

def findEpsParams(features):
    
    neigh = NearestNeighbors(n_neighbors=NEIGHBOURS)
    nbrs = neigh.fit(features)
    distances, indices = nbrs.kneighbors(features)
    
    print(distances)

    distances = np.sort(distances, axis=0)[:,NEIGHBOURS-1]
    plt.plot(distances)
    plt.show()

    return 0.3

if __name__ == "__main__":
    print("start main")
    main()
    print("end main")