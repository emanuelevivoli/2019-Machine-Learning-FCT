# libraries for Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap

# libraries for general utilities
import numpy as np 
from pandas.plotting import scatter_matrix
from pandas import DataFrame

def features_creation(dataset, NUMBER_FEATURES = 6):
    """
        FEATURES CREATION
        
    """
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

    return features


def importance_test_SVM(features, labels, title='importance_test_SVM'):
        """
        FEATURES IMPORTANCE GRAPH using SVM and different scale method such
        - NORMALIZATION
        - STANDARDIZATION
        - MIN MAX NORMALIZATION

        Univariate feature selection with F-test for feature scoring
        We use the default selection function to select the
        most significant features
    """
    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split

    feat = features[labels!=0]
    lbs = labels[labels!=0]

    k_selection = 'all'

    # Split dataset to select feature and evaluate the classifier
    X_train, X_test, y_train, y_test = train_test_split(
            feat, lbs, stratify=lbs, random_state=4
    )

    selector = SelectKBest(f_classif, k=k_selection)
    selector.fit(X_train, y_train)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.sum()
    X_indices = np.arange(feat.shape[-1])
    plt.bar(X_indices, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)', color='darkorange')#,edgecolor='black')

    # #############################################################################
    # Compare to the weights of an SVM
    clf = make_pipeline(MinMaxScaler(), LinearSVC())
    clf.fit(X_train, y_train)
    print('Classification accuracy without selecting features: {:.3f}'
        .format(clf.score(X_test, y_test)))

    svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
    svm_weights /= svm_weights.sum()

    plt.bar(X_indices + .2, svm_weights, width=.2, label='SVM weight',
            color='navy')#,edgecolor='black')

    # #############################################################################
    # SVM on Normalized input

    clf_selected = make_pipeline(
            SelectKBest(f_classif, k=k_selection), #MinMaxScaler(), 
            Normalizer(),
        LinearSVC()
    )
    clf_selected.fit(X_train, y_train)
    print('Classification accuracy after selection (Normalized): {:.3f}'
        .format(clf_selected.score(X_test, y_test)))

    svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.sum()

    plt.bar(X_indices[selector.get_support()] + .4, svm_weights_selected,
            width=.2, label='SVM selection (Normalized)', color='c')#,edgecolor='black')


    # #############################################################################
    # SVM on Standardize input

    clf_selected = make_pipeline(
            SelectKBest(f_classif, k=k_selection), #MinMaxScaler(), 
            StandardScaler(),
        LinearSVC()
    )
    clf_selected.fit(X_train, y_train)
    print('Classification accuracy after selection (Standardize): {:.3f}'
        .format(clf_selected.score(X_test, y_test)))

    svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.sum()

    plt.bar(X_indices[selector.get_support()] + .6, svm_weights_selected,
            width=.2, label='SVM selection (Standardize)', color='pink')#,edgecolor='black')


    # #############################################################################
    # SVM on MinMax Normalized input

    clf_selected = make_pipeline(
            SelectKBest(f_classif, k=k_selection), 
        MinMaxScaler(),
        LinearSVC()
    )
    clf_selected.fit(X_train, y_train)
    print('Classification accuracy after selection (MinMax norm): {:.3f}'
        .format(clf_selected.score(X_test, y_test)))

    svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.sum()

    plt.bar(X_indices[selector.get_support()] + .8, svm_weights_selected,
            width=.2, label='SVM selection (MinMax norm)', color='red')#,edgecolor='black')

    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')

    plt.savefig(r"imgs/{title}.png")
    


def features_standardization(features, labels):
    """
        FEATURES STANDARDIZATION
        
    """
    scaler = StandardScaler()
    features_std = scaler.fit_transform(features)
    ids = labels[:,0]
    labels = labels[:,1]

    return features_std, ids, labels

def features_correlation_matrix(features, title='correlation_matrix'):
    """
        CREATES CORRELATION MATRIX
        
    """
    corr = abs(DataFrame(features).corr())
    corr.style.background_gradient(cmap='coolwarm').format("{:.3}")
    plt.savefig(r"imgs/{title}.png")
    return corr

def correlated_features(corr, CORRELATION_LIMIT=0.6):
    """
        CALCULATES FEATURE MORE CORRELATED
        
    """
    (xs, ys) = np.where((corr>CORRELATION_LIMIT)==True)
    pairs = [ [x, y] for x,y in zip(xs, ys) if x>y ]
    indexes = np.array([int(x) for x in np.union1d(np.ravel(pairs),[])])
    return indexes

def correlated_scatter_matrix(features_std, indexes, title='correlated_scatter_matrix'):
    """
        SAVES SCATTER MATRIX IMAGE
        
    """

    Axes = scatter_matrix(DataFrame(features_std[:,indexes], columns=indexes), alpha=0.5, diagonal='kde')
    #y ticklabels
    [plt.setp(item.yaxis.get_majorticklabels(), 'size', 5) for item in Axes.ravel()]
    #x ticklabels
    [plt.setp(item.xaxis.get_majorticklabels(), 'size', 5) for item in Axes.ravel()]
    #y labels
    [plt.setp(item.yaxis.get_label(), 'size', 13) for item in Axes.ravel()]
    #x labels
    [plt.setp(item.xaxis.get_label(), 'size', 13) for item in Axes.ravel()]

    plt.savefig(r"imgs/{title}.png")


def shapiro_ranking(final_feat, labels, title='shapiro_ranking'):
    from yellowbrick.datasets import load_credit
    from yellowbrick.features import Rank1D

    # Instantiate the 1D visualizer with the Sharpiro ranking algorithm
    visualizer = Rank1D(algorithm='shapiro')

    visualizer.fit(final_feat[labels!=0],labels[labels!=0] )           # Fit the data to the visualizer
    visualizer.transform(final_feat[labels!=0])        # Transform the data
    # visualizer.show()              # Finalize and render the figure

    plt.savefig(r"imgs/{title}.png")


def pearson_ranking(final_feat, labels, title='pearson_ranking'):

    from yellowbrick.features import Rank2D

    # Instantiate the visualizer with the Pearson ranking algorithm
    visualizer = Rank2D(algorithm='pearson')

    fig = plt.figure(i, figsize=(8, 6))
    i = i+1

    visualizer.fit(final_feat[labels!=0],labels[labels!=0] )           # Fit the data to the visualizer
    visualizer.transform(final_feat[labels!=0])        # Transform the data
    
    plt.savefig(r"imgs/{title}.png")
    plt.close()


# radvits_data(final_feat[labels!=0,:], labels[labels!=0], title='labeled')
# radvits_data(final_feat, labels, title='all')
def radvits_data(final_feat, labels, title='labeled'):
    df = DataFrame({
            '1 feature': final_feat[:,0],
            '2 feature': final_feat[:,1],
            '3 feature': final_feat[:,2],
            '4 feature': final_feat[:,3],
            'Category': lab
    })
    rad_viz = radviz(df, 'Category')

    plt.savefig(r"imgs/radvits_data_{title}.png")
    plt.close()


def remove_features(features, indexes):
    """
        FEATURES REMOVING
    """

    b = np.arange(features.shape[1])
    b = np.setdiff1d(b,np.array(pairs)[:,0])

    features_sel = features[:,b]

    return features_sel


def anova_f_test_selection(features, labels, alpha=0.05):
    """
        FEATURES EXTRACTION
    """

    from scipy.stats import f as f_func

    # selection with ANOVA F-test
    f, prob = f_classif(features, labels)

    # extract importance features indexes from f values
    app = [ [i, ele] for i, ele in enumerate(f)]
    app.sort(key = lambda app: app[1], reverse = True)
    indexes = np.array([ int(a) for a in np.array(app)[:,0]])

    # calculating values for the f distribution critic value
    p = len(indexes)
    n = len(labels)

    # calculating the critic value
    critical_value = f_func.ppf(q=1-alpha, dfn=p-1, dfd=n-p)

    # retire only feature that reject the H0 hypothesis
    retire = f > critical_value

    return features[:,indexes[retire]]
    

