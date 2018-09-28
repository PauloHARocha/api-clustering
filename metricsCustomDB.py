import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans
from algorithms.metrics import Metrics


def execMetricsDatasets(datasets, algorithm, k, metrics, n_sim): 
    rs_metrics = []
    rs_datasets = []
    mets_results = {}
    aux_results = {}
    
    for met in metrics:
        mets_results[met] = []
    
    for ds in range(len(datasets)):
        rs_datasets.append({'name': ds, 'centroids': [], 'clusters': [], })

    for sim in tqdm(range(n_sim)):
        for met in metrics:
            aux_results[met] = []
        for idx, dataset in enumerate(datasets):
            ag_exec = algorithm(data=dataset)
            ag_exec.fit(k=k)
            clusters = ag_exec.clusters
            centroids = ag_exec.centroids
            
            for met in tqdm(metrics, desc='dataset: {}'.format(idx)):
                aux_results[met].append(Metrics.evaluate(
                    met, dataset, centroids, clusters, algorithm, k))
                
            rs_centroids, rs_clusters = prepareToListResponse(centroids, clusters)

            for ds in range(len(datasets)):
                if rs_datasets[ds]['name'] == idx:
                    rs_datasets[ds]['centroids'].append(rs_centroids)
                    rs_datasets[ds]['clusters'].append(rs_clusters)

        for met in metrics:
            mets_results[met].append(aux_results[met])
    
    for met in metrics:
        rs_metrics.append({'name': met, 'values': mets_results[met]})

    response = { 'metrics': rs_metrics,'datasets': rs_datasets, }
    
    return response


def generate_metrics_datasets(algorithm_id, k, ds_idx):
    algorithms = [KMeans, FCMeans]

    metrics = ['inter-cluster', 'cluster-separation', 'abgss', 
               'edge-index', 'cluster-connectedness', 'intra-cluster',
               'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
               'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
               'silhouette', 'min-max-cut', 'gap']

    custom_ds = []
    for idx in range(2): #12
        #Importing dataset
        ds = pd.read_csv("custom_datasets/{}/dataBase_{}.csv".format(ds_idx,idx))
        #Select lines and columns 
        custom_ds.append(ds.iloc[:, [1, 2]].values)
    
    algorithm = algorithms[algorithm_id]

    met_db_results = execMetricsDatasets(
        datasets=custom_ds, algorithm=algorithm, k=k, metrics=metrics, n_sim=2)

    return met_db_results


def prepareToListResponse(centroids, clusters):
    for c in centroids:
        centroids[c]=centroids[c].tolist()

    for c in clusters:
        for xi in range(len(clusters[c])):
            clusters[c][xi]=clusters[c][xi].tolist()
    
    rs_centroids = []
    rs_clusters = []
    for c in centroids:
        rs_centroids.append({'name': c, 'values': centroids[c]})
    for c in clusters:
        rs_clusters.append({'name': c,'values': clusters[c]})

    return rs_centroids, rs_clusters
