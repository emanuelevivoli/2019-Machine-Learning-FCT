# First Assignment

This project is composed by various files.
The files were required from the assignment:
* **TP1.txt**
    - This is the questions and answers file.
* **TP1.py**
    - This is a Python 3.x script that can be used to run your code for this assignment, it contains the main function.
* **NB.png**
    - The plot of training and cross-validation errors for the KDE kernel width of your implementation of the Naïve Bayes classifier.
* **SVM.png**
    - The plot of training and cross-validation errors for the gamma parameter of the SVM classifier.

Some additional files that contains classes and functions:
* **utils.py**
    - This is a Python 3.x script that contains all the classes definitions and functions used in TP1.py.
* **NaiveBayes.py**
    - This is a Python 3.x script that contains the our implementation of NaiveBayes.
* **evaluation.csv**
    - This files is (eventually) saved by the hyperparameter optimization of C and gamma, used as log file.

And two other files with the instructions:
* **ASSIGNMENT.md**
    - The file with the rules for the assignment that you can also find at [link course](http://aa.ssdi.di.fct.unl.pt/tp1.html)
* **README.md**
    - This file, with the **How to use** explanation.

In order to make all the process (the load of the sets too) there are also the train and test sets files:
* **TP1_train.tsv**
    - This is the training data set. Use this to optimize parameters and train the classifiers.
* **TP1_test.tsv**
    - This is the test set for estimating the true error and comparing the final classifiers.

## Objective

The goal of this assignment is to parametrize, fit and **compare Naive Bayes and Support Vector Machine classifiers**. The data set is inspired on the banknote authentication problem in the [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication), but the data was adapted for this assignment. You can find in this repo the downloaded data files called:

* **TP1_train.tsv**
    - This is the training data set. Use this to optimize parameters and train the classifiers.
* **TP1_test.tsv**
    - This is the test set for estimating the true error and comparing the final classifiers.

## System Requirements

* numpy
* sklearn
* matplotlib
* intertools

And (facultative) only for hyper-parameter optimization, you will need also:
* csv 
* rbfopt 
* bonmin, download and install from [here](https://ampl.com/products/solvers/open-source/). You will need to remember the path where you have installed it (like `"/path/to/bonmin"`)

## How to use

To run the experiment and see the result you just need to run:
```bash
$ python TP1.py
```

In case you want to use the Hyperparameter optimization for SVM (for parameters C and gamma), and only in case you have installed rbfopt and bonmin, you can use:
```bash
$ python TP1.py --hyper-opt "/path/to/bonmin" --opti-steps 50 
```
You have also 2 other parameters that you can pass to the main file in order to replicate experiments:
* set the seed
    ```bash
    $ python TP1.py --seed 30
    ```
* set the k-fold split number
    ```bash
    $ python TP1.py --n-split 5
    ```