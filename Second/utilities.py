from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
import numpy as np 

#date le labels iniziali
#ritorna la lista degli indici!=0 e la lista delle relative classi
def take_valid_labels(labels):
    valid_labels = []
    valid_class = []
    not_valid_labels = []
    ind=0

    for el in labels:
        if el[1]!=0:
            valid_labels.append(ind)
            valid_class.append(el[1])
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
    
#date le feature, il k massimo, l'indici !=0 delle label iniziali
#ritorna i valori silhouette e ars per ogni k
def kmeans_all_results(featu, kmax, vlarray):
    int_index_values = []
    ext_index_values = []
    
    for k in range(2, kmax+1):
        klabels = KMeans(n_clusters = k).fit_predict(featu)
        int_index_values.append([k,silhouette_score(featu, klabels, metric = 'euclidean')])
        v_labels = take_selected_from_array(klabels,vlarray[0].tolist())
        ext_index_values.append([k,adjusted_rand_score(vlarray[1],v_labels)])

    return int_index_values,ext_index_values