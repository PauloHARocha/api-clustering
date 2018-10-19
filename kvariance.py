import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from algorithms.metrics import Metrics
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans

def generate_kvariance(dataset_id, algorithm_id, k_min, k_max, n_sim):
    
    algorithms = [KMeans, FCMeans]

    metrics = ['inter-cluster', 'cluster-separation', 'abgss',
               'edge-index', 'cluster-connectedness', 'intra-cluster',
               'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
               'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
               'silhouette', 'min-max-cut', 'gap']

    
    if dataset_id == 3:
        dataset = pd.read_csv("custom_datasets/dataPhDAlzheimerSemNomes.csv")
        dataset_labeled = dataset.iloc[:, [3, 4, 5, 6, 9]].values
        dataset = dataset.iloc[:, [3, 4, 5, 6]].values
        labels_names = [1, 10, 100]
        normalize = False
    elif dataset_id == 4:
        dataset = pd.read_csv("custom_datasets/dataPhDAlzheimerSemNomes.csv")
        dataset_labeled = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 9]].values
        dataset = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values
        labels_names = [1, 10, 100]
        normalize = False
    else:
        ds = [datasets.load_iris(), datasets.load_wine(),
              datasets.load_diabetes()]
        labels = [[0, 1, 2], [0, 1, 2]]
        dataframe = pd.DataFrame(ds[dataset_id].data[:, :])
        dataset = dataframe.values
        
        dataframe['CLASS'] = ds[dataset_id].target
        dataset_labeled = dataframe.values
        labels_names = labels[dataset_id]
        normalize = True

    algorithm = algorithms[algorithm_id]
    k_rng = range(k_min, k_max+1)
    response = execMetrics(
        dataset=dataset, ds_labeled=dataset_labeled, labels_names=labels_names,
        algorithm=algorithm, k_rng=k_rng, metrics=metrics, 
        n_sim=n_sim, k_min=k_min, normalize=normalize)

    #Write scenarios
    file_name = 'scenarios/ds{}_ag{}_k{}-{}_sim{}.json'.format(
        dataset_id, algorithm_id, k_min, k_max, n_sim)
    with open(file_name, 'w') as outfile:
        json.dump(response, outfile)

    return response


def execMetrics(dataset, ds_labeled, algorithm, labels_names,k_rng, metrics, n_sim, k_min, normalize=True):

    mets_results = {}
    aux_metrics = {}
    rs_centroids = []
    rs_clusters = []

    if normalize:
        std = MinMaxScaler()
        dataset = std.fit_transform(dataset)

    for met in metrics:
        mets_results[met] = []
    for sim in tqdm(range(n_sim), desc='sim'):
        
        sim_centroids = []
        sim_clusters = []
        
        for met in metrics:
            aux_metrics[met] = []
        for k in tqdm(k_rng, desc='k'):
            ag_exec = algorithm(data=dataset)
            ag_exec.fit(k=k)
            clusters = ag_exec.clusters
            centroids = ag_exec.centroids
            for met in metrics:
                aux_metrics[met].append(Metrics.evaluate(
                    met, dataset, centroids, clusters, algorithm, k))
            
            conf_matrix = generate_conf_matrix(ag_exec, dataset, ds_labeled, labels_names)

            centroids, clusters = prepareToList(centroids, clusters)
            
            aux_centroids = []
            aux_clusters = []
            for cent in centroids:
                aux_centroids.append({ 'name': cent, 'values': centroids[cent] })
            
            for clust in clusters:
                aux_clusters.append({ 'name': clust, 'values': clusters[clust] })
            
            sim_centroids.append(aux_centroids)
            sim_clusters.append(aux_clusters)

        rs_centroids.append(sim_centroids)
        rs_clusters.append(sim_clusters)

        for met in metrics:
            mets_results[met].append(aux_metrics[met])

    rs_metrics = []
    for met in mets_results:
        rs_metrics.append({'name': met, 'values': mets_results[met]})

    response = {'centroids': rs_centroids, 'clusters': rs_clusters,
                'metrics': rs_metrics, 'k_min': k_min, 'conf_matrix': conf_matrix}
    return response


def prepareToList(centroids, clusters):
    for c in centroids:
        centroids[c]=list(centroids[c])

    for c in clusters:
        for xi in range(len(clusters[c])):
            clusters[c][xi]=list(clusters[c][xi])
    
    return centroids, clusters

def generate_conf_matrix(ag_exec, dataset, ds_labeled, labels_names):
    class_ = []
    for x in dataset:
        class_.append(ag_exec.predict(x))
    conf_df = pd.DataFrame(ds_labeled)
    conf_df['CLASS'] = class_

    y_true = conf_df.iloc[:,-2].values
    y_pred = conf_df.iloc[:,-1].values

    
    k_count = {}
    for k in range(ag_exec.k):
        k_count[k] = {}
        for lb in labels_names:
            k_count[k][lb] = 0

    for k in range(ag_exec.k):
        for true, pred in zip(y_true, y_pred):
            if pred == k:
                k_count[k][true] += 1
    
    for k in k_count:
        label_count = 0
        new_label_name = ''
        for l in k_count[k]:
            if k_count[k][l] > label_count:
                label_count = k_count[k][l]
                new_label_name = l
        for pred in range(y_pred.size):
            if y_pred[pred] == k:
                y_pred[pred] = new_label_name*(-1)
    
    for pred in range(y_pred.size):
        y_pred[pred] = y_pred[pred]*(-1)
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    cf_json = {}
    for row, label in zip(conf_matrix, labels_names):
        cf_json[label] = row.tolist()
    
    return cf_json



    
