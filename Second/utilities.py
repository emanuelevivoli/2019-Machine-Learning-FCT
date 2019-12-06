from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
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
    
#date le feature, il k massimo, l'indici !=0 delle label iniziali
#ritorna i valori silhouette e ars per ogni k
def kmeans_all_results(featu, kmax, vlarray):
    int_index_values = []
    ext_index_values = []
    klabels_for_k = []
    
    for k in range(2, kmax+1):
        klabels = KMeans(n_clusters = k).fit_predict(featu)
        klabels_for_k.append(klabels)
        int_index_values.append([k,silhouette_score(featu, klabels, metric = 'euclidean')])
        v_labels = take_selected_from_array(klabels,vlarray[0].tolist())
        ext_index_values.append([k,adjusted_rand_score(vlarray[1],v_labels)])

    return int_index_values,ext_index_values,klabels_for_k

'''
def BiKMemans_predict(kmax,featu):
    featu_dict=dict()
    prof_results=dict()

    for i in range(0,563):
            #print(i)
            featu_dict[i]=featu[i]
            prof_results[i]=list([])

    results=dict()
    kmax_ind=0

    for k in range (2,kmax+1):
        #featu.dict
        valu_list=featu_dict
        #print(valu_list.values())
        valu_array= np.array(list(valu_list.values()))
        index_array= np.array(list(valu_list.keys()))
        #print(valu_array)
        index_dict=dict()

        print("ia",index_array)

        klabels = KMeans(2).fit_predict(valu_array)
        n_to_add_labels= ((kmax_ind)*2)
        #print(klabels,"K")

        klabels=klabels+n_to_add_labels
        print("K2",klabels)
        first_cluster=dict()
        second_cluster=dict()
        temp_results=dict()

        for i in range(0,len(klabels)):

            realIndex = index_array[i]
            temp_results[realIndex]=klabels[i]
            if klabels[i]==n_to_add_labels:
                first_cluster[realIndex]=featu_dict[realIndex]
            else:
                second_cluster[realIndex]=featu_dict[realIndex]
            prof_label=klabels[i]-n_to_add_labels    
            prof_results[realIndex].append(prof_label)

        if len(first_cluster)<len(second_cluster):
            print("f")
            for el,val in first_cluster.items():
                print(el," label: ",temp_results[el])
                results[el]=temp_results[el]

            if k==kmax:
                print("F FINE")
                for el,val in second_cluster.items():
                    #print(el,val)
                    print(el)
                    results[el]=temp_results[el]
            else:
                featu_dict = second_cluster.copy()
        else: 
            print("s")
            for el,val in second_cluster.items():
                print(el," label: ",temp_results[el])
                results[el]=temp_results[el]


            if k==kmax:
                print("S FINE")
                for el,val in first_cluster.items():
                    #print(el,val)
                    print(el)
                    results[el]=temp_results[el]
            else:
                featu_dict = first_cluster.copy()


        first_cluster.clear()
        second_cluster.clear()
        kmax_ind=kmax_ind+1

    print(results)

    list_result=[]
    for k, v in results.items(): 
        #print(k, v)
        list_result.append(v)
    print(list_result)
    print(prof_results)
    return list_result,list(prof_results.values())
'''

def select_two_empty_dictionary(d):

    first=-1
    for di in d:
        print(di)
        if(not di):
            print("void d: ",di)
            if first==-1:
                first=di.copy()
            else:
                return first,di.copy()
    return {"error"},{"error"}

def get_larger_dictionary(d):
    larger_one = dict()
    larger_one[0]=0
    for di in d:
        print(di)
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