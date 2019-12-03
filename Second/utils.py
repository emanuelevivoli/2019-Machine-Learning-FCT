import numpy as np
from tp2_data.tp2_aux import *

def set_seed(seed):
    global SEED
    SEED = None if seed == None else int(seed)

class Loader:
    
    def __init__(self):

        self.dataset = np.empty
        self.labels = np.empty
        self.class_split_dataset = np.empty

    def load(self, file_name="./tp2_data/labels.txt"):

        # take data from the file and put all in a ndArray
        mat = images_as_matrix(folder='tp2_data/')
        labels = np.loadtxt(file_name, delimiter=',')
        
        # split dataset for class label [0, 1, 2, 3]
        self.class_split_dataset = np.array([ mat[labels[:,1]==class_type,:] for class_type in set(labels[:,1])])

        # create an indexes array
        indexes = np.arange(start=0, stop=len(labels), step=1)

        # randomize the indexes
        np.random.seed(SEED)
        np.random.shuffle(indexes)

        # labels take only the labels column
        # dataset take the data without labels
        self.dataset = mat[indexes]
        self.labels = labels[indexes]

        return self

    def getDataset(self):
        return self.dataset

    def getLabels(self):
        return self.labels

    def getClassSplitDataset(self):
        return self.class_split_dataset