##
## DBSCAN
##

# our libraries
import utilities as uti

import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from mpl_toolkits.mplot3d import Axes3D

START_FEAT = 2
MARGIN = 15
NEIGHBOURS = 5

def find_eps_params(features, title="find_eps_params"):

    num_features = features.shape[1]
    
    neigh = KNeighborsClassifier(n_neighbors=NEIGHBOURS)
    nbrs = neigh.fit(features, np.ones(features.shape[0]))
    distances, indices = nbrs.kneighbors(features)

    #print(distances)

    distances = np.sort(distances, axis=0)[:,NEIGHBOURS-1]
    # plt.plot(np.arange(len(distances)), distances, '--', c='red')

    z = np.polyfit(np.arange(len(distances)), distances, 15)
    f2 = np.poly1d(z)
    df2 = np.polyder(f2,1)

    xnew = np.linspace(0, len(distances),features.shape[0], endpoint=True)

    #print(xnew)
    y = f2(xnew)
    ynew = np.array([50*e for e in df2(xnew)])

    #print(ynew.shape)
    x_index = np.where(np.r_[True, ynew[1:] < ynew[:-1]] & np.r_[ynew[:-1] < ynew[1:], True] == True)[0][-3:]
    x_index[-1]=x_index[-1] + MARGIN
    print(x_index)

    plt.axhline(y=y[x_index[0]], linestyle='--', c='blue')
    plt.axhline(y=y[x_index[-1]], linestyle='--', c='red')

    plt.axvline(x=x_index[0], linestyle='--', c='blue')
    plt.axvline(x=x_index[-1], linestyle='--', c='red')

    plt.plot( xnew, y, '-', c='black')

    plt.savefig(f"imgs/find_eps_params/{title}_{num_features}_feats.png")

    return (y[x_index[0]], y[x_index[-1]])


def DBSCAN_clustering(final_feat, labels, title="find_eps_params"):

    # clustering with DBSCAN
    dbscan_by_num = []
    eps_matrix = []

    for numb in np.arange(START_FEAT,final_feat.shape[1],1):
        dbscan_by_eps = []
        
        # automatically selecting epsilon
        eps_range = find_eps_params(final_feat[:,:numb], title)
        
        eps_matrix.append(eps_range)
        
        for eps in np.linspace(eps_range[0], eps_range[1], 50):

            dbscan = DBSCAN(eps=eps, min_samples=NEIGHBOURS)
            dbscan_by_eps.append(dbscan.fit_predict(final_feat[:,:numb])[labels!=0])
            
        dbscan_by_num.append(dbscan_by_eps)

    return np.array(dbscan_by_num), np.array(eps_matrix)


def calculate_scores(labels, final_feat, dbscan_by_num):

    score_by_num = []

    #removing class 0 in the original labels
    lab = np.array(labels[labels!=0])
    lab = np.array([int(l) for l in lab])


    for numb in np.arange(final_feat[labels!=0].shape[1]-START_FEAT):
        score_by_eps = []
        for eps in np.arange(50):

            app_lab = lab            
            app_clu = dbscan_by_num[numb][eps]
            
            # change cluster -1 to max+1
            mx = np.max(app_clu)
            mn = np.min(app_clu)
            app_clu[app_clu==mn] = mx+1
            
            # scores
            #(tp, fp, fn, tn) = uti.scores(app_clu, app_lab)
            #randm = uti.rand_index_score(tp, fp, fn, tn) 
            #preci = uti.precision(tp, fp, fn, tn)
            #recal = uti.recall(tp, fp, fn, tn)
            #effe1 = uti.f1(preci, recal)
            #score_by_eps.append([randm, preci, recal, effe1])
            
            adjrn = adjusted_rand_score(app_clu, app_lab)
            score_by_eps.append([adjrn])

            #adjrn = adjusted_rand_score(app_clu, app_lab)
            #silue = silhouette_score(app_clu, app_lab, metric = 'euclidean')
            #score_by_eps.append([randm, preci, recal, effe1, adjrn])
            
        score_by_num.append(score_by_eps)
        
    return np.array(score_by_num)


def calculate_bests_indexes(score_by_num):

    win = np.nanmax(score_by_num, (0,1))
    win_ids = []
    skip = []
    for k in win:
        were = np.where(score_by_num == k)
        if not (np.ravel(were).size/3 >  score_by_num.shape[1]/3):
            win_ids.append(were)
        else:
            win_ids.append(False)

    win_ids = np.array( [ [ (row[0][j], row[1][j], i) for j, x in enumerate(row[2]) if x==i ] for i, row in enumerate(win_ids) if row != False])
    
    return win_ids


def indexes_to_dict(win_ids):

    win_dic = dict() 
    for ele in win_ids:
        for x, y, z in ele:
            if (x,y) not in win_dic:
                win_dic[(x,y)] = set()
            win_dic[(x,y)].add(z)
    return win_dic


def save_3d_plot(win_dic, eps_matrix, dbscan_by_num, final_feat,  title='cluster'):

    for x, y in win_dic.keys():
        print(x,y)
        
        num_features = x

        (idi, idj) = list(win_dic.keys())[0]
        a = eps_matrix[idi]
        epsi = np.linspace(a[0], a[1], 50)[36]

        print(win_dic[(x,y)],'\n',dbscan_by_num[x,y])
        
        fig = plt.figure()
        ax = Axes3D(fig, elev=-150, azim=110)

        ff = final_feat[:, :x+1]
        db = dbscan_by_num[x,y]

        #ff = ff[db!=-1,:]
        #db = db[db!=-1]

        print(np.unique(db))

        ax.scatter(ff[:, 0], ff[:, 1], ff[:, 2], c=db,
                cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title(f"Clustering with DBSCAN [{num_features},{epsi}]")
        ax.set_xlabel("1st feature")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd feature")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd feature")
        ax.w_zaxis.set_ticklabels([])

        plt.savefig(f"imgs/clusters/{title}_{num_features}_feats_{np.round(epsi,3)}_epsi.png")
        plt.close()