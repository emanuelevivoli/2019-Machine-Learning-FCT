from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
from utils import *
import numpy as np 

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb  # noqa 

def scores(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp, fp, fn, tn)

def rand_index_score(tp, fp, fn, tn):
    return (tp + tn) / (tp + fp + fn + tn)

def precision(tp, fp, fn, tn):
    return tp/(fp + tp)

def recall(tp, fp, fn, tn):
    return tp/(fn + tp)

def f1(precision, recall):
    return 2 * (precision * recall)/(precision + recall)

#date le labels iniziali
#ritorna la lista degli indici!=0 e la lista delle relative classi
def take_valid_labels(labels):
    valid_labels = []
    valid_class = []
    not_valid_labels = []
    ind=0

    for el in labels:
        if el!=0:
            valid_labels.append(ind)
            valid_class.append(el)
        else:
            not_valid_labels.append(ind)
        ind=ind+1

    np.array(valid_labels)        
    vlarray = np.array(valid_labels) 
    clarray = np.array(valid_class)
    #nlarray = np.array(not_valid_labels)
 
    return vlarray,clarray

# ritorna un array con solo gli el di indice presente in valid_values
def take_selected_from_array(array,valid_values):
    res=[]
    for i in valid_values:
        res.append(array[i])
    return res

def mixture_all_results(featu,kmin, kmax, vlarray):
    int_index_values = []
    ext_index_values = []
    klabels_for_k = []
    
    for k in range(kmin, kmax+1):
        klabels = mixture.GaussianMixture(n_components = k, covariance_type='full').fit_predict(featu)
        klabels_for_k.append(klabels)
        int_index_values.append([k,silhouette_score(featu, klabels, metric = 'euclidean')])
        v_labels = take_selected_from_array(klabels,vlarray[0].tolist())
        ext_index_values.append([k,adjusted_rand_score(vlarray[1],v_labels)])

    return int_index_values,ext_index_values,klabels_for_k
    
#date le feature, il k massimo, l'indici !=0 delle label iniziali
#ritorna i valori silhouette e ars per ogni k
def kmeans_all_results(featu,kmin, kmax, vlarray):
    int_index_values = []
    ext_index_values = []
    klabels_for_k = []
    
    for k in range(kmin, kmax+1):
        klabels = KMeans(n_clusters = k).fit_predict(featu)
        klabels_for_k.append(klabels)
        int_index_values.append([k,silhouette_score(featu, klabels, metric = 'euclidean')])
        v_labels = take_selected_from_array(klabels,vlarray[0].tolist())
        ext_index_values.append([k,adjusted_rand_score(vlarray[1],v_labels)])

    return int_index_values,ext_index_values,klabels_for_k

def kmeans_all_results_for_different_features(featu,indexes,n_cluster_max,n_feat_max,vlarray):
    res=dict();
    for n_feat in range(2,n_feat_max):
        features_last_sel = features_sel[:,indexes[:n_feat]]
        featu=features_last_sel
        res[n_feat]= uti.kmeans_all_results(featu,n_cluster_max,vlarray)
    return res

def select_two_empty_dictionary(d):

    first=-1
    for di in d:
        #print(di)
        if(not di):
            #print("void d: ",di)
            if first==-1:
                first=di.copy()
            else:
                return first,di.copy()
    return {"error"},{"error"}

def get_larger_dictionary(d):
    larger_one = dict()
    larger_one[0]=0
    for di in d:
        #print(di)
        if len(larger_one) < len(di):
            larger_one=di.copy()
    return larger_one

def new_BiKMemans_predict(kmax,featu):
    featu_dict=dict()
    prof_results=dict()
    cluster_list=[]

    for i in range(0,563):
        featu_dict[i]=featu[i]
        prof_results[i]=list([])
            
    kmax_ind=0

    for k in range (2,kmax+1):
        
        valu_list=featu_dict
        
        #chiavi
        index_array= np.array(list(valu_list.keys()))
        #valori
        valu_array= np.array(list(valu_list.values()))
        
        index_dict=dict()
        
        klabels = KMeans(2).fit_predict(valu_array)
        
        first_cluster=dict()
        second_cluster=dict()
        
        for i in range(0,len(klabels)):

            realIndex = index_array[i]
            
            if first_cluster != -1 and second_cluster != -1:
                if klabels[i]==0:
                    first_cluster[realIndex]=featu_dict[realIndex]
                else:
                    second_cluster[realIndex]=featu_dict[realIndex]
                
                prof_label=klabels[i]    
                prof_results[realIndex].append(prof_label)
                
        cluster_list.append(first_cluster)
        cluster_list.append(second_cluster)
        
        larger=get_larger_dictionary(cluster_list)
        featu_dict=larger.copy() 
        cluster_list.remove(larger)

        kmax_ind=kmax_ind+1

    
    return list(prof_results.values())


#date le feature, il k massimo, l'indici !=0 delle label iniziali
#ritorna i valori silhouette e ars per ogni k
def bi_kmeans_all_results(featu, kmax, vlarray):
    int_index_values = []
    ext_index_values = []
    klabels_for_k = []
    groups_for_bk = []
    
    for k in range(2, kmax+1):
        prof_result = NewBiKMemans_predict(k,featu)
        groups_for_bk.append(prof_result)
        int_index_values.append([k,silhouette_score(featu, klabels, metric = 'euclidean')])
        v_labels = take_selected_from_array(klabels,vlarray[0].tolist())
        ext_index_values.append([k,adjusted_rand_score(vlarray[1],v_labels)])

    return int_index_values,ext_index_values,klabels_for_k, groups_for_bk


def aggl_all_results(featu,kmin, kmax, vlarray):
    int_index_values = []
    ext_index_values = []
    klabels_for_k = []
    
    for k in range(kmin, kmax+1):
        klabels =  AgglomerativeClustering(n_clusters = k).fit_predict(featu)
        klabels_for_k.append(klabels)
        int_index_values.append([k,silhouette_score(featu, klabels, metric = 'euclidean')])
        v_labels = take_selected_from_array(klabels,vlarray[0].tolist())
        ext_index_values.append([k,adjusted_rand_score(vlarray[1],v_labels)])

    return int_index_values,ext_index_values,klabels_for_k

def clustering_valutation_visualization(file_prefix,ids,labels,featu,kmin,n_cluster_max,f1):
    
    vlarray=take_valid_labels(labels)
    
    n_feat=featu.shape[1]
    int_aggl, ext_aggl, k_labels = f1(featu,kmin,n_cluster_max,vlarray)
    int_aggl = np.array(int_aggl)
    int_aggl[np.where(int_aggl == np.max(int_aggl, 0)[1])[0][0]][0]
    best_int_k=int(int_aggl[np.where(int_aggl == np.max(int_aggl, 0)[1])[0][0]][0])
    print("the best k, according to the internal index is: ",best_int_k)
    first_file_name="tp2_data/"+file_prefix+"_"+str(best_int_k)+"cluster"+str(n_feat)+"feat_int.html"
    report_clusters(ids, k_labels[best_int_k-kmin], first_file_name)
    print("Visualization in the file: ",first_file_name)
    
    ext_aggl = np.array(ext_aggl)
    best_ext_k=int(ext_aggl[np.where(ext_aggl == np.max(ext_aggl, 0)[1])[0][0]][0])
    print("the best k, according to the external index is: ",best_ext_k)
    if best_ext_k!= best_int_k:
        second_file_name= "tp2_data/"+file_prefix+"_"+str(best_ext_k)+"cluster"+str(n_feat)+"feat_ext.html"
        report_clusters(ids, k_labels[best_ext_k-kmin], second_file_name)
        print("Visualization in the file: ",second_file_name)
    
    return int_aggl, ext_aggl, k_labels
    
    

def bisect_kmeans_visualization(ids,featu,n_feat,n_cluster):
    featu=featu[:,:n_feat]
    res_real_bi=new_BiKMemans_predict(n_cluster,featu)
    file_name="tp2_data/NewBiKmeans"+str(n_cluster)+"cluster"+str(n_feat)+"feat.html"
    print("Visualization in the file: ",file_name)
    report_clusters_hierarchical(ids,res_real_bi,file_name)
    