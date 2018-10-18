import numpy as np
import json
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans
from algorithms.metrics import Metrics


def execMetrics(dataset, algorithm, k, metrics, n_sim, normalize=True):

    mets_results = {}
    rs_centroids = []
    rs_clusters = []
    
    if normalize:
        std = MinMaxScaler()
        dataset = std.fit_transform(dataset)

    for met in metrics:
        mets_results[met] = []

    ag_exec = algorithm(data=dataset)
    for sim in tqdm(range(n_sim)):    
        ag_exec.fit(k=k)

        aux_results = {}
        for met in metrics:
            aux_results[met] = []
            for itr in tqdm(range(len(ag_exec.all_centroids)), desc='{}'.format(met)):
                clusters = ag_exec.all_clusters[itr]
                centroids = ag_exec.all_centroids[itr]
                
                aux_results[met].append(Metrics.evaluate(
                    met, dataset, centroids, clusters, algorithm, k))

        for met in metrics:
            mets_results[met].append(aux_results[met])

        centroids = ag_exec.all_centroids
        clusters = ag_exec.all_clusters
        aux_centroids = []
        aux_clusters = []
        for itr in ag_exec.all_centroids:
            itr_centroids = []
            itr_clusters = []

            centroids[itr], clusters[itr] = prepareToList( centroids[itr], clusters[itr])
            
            for cent in centroids[itr]:
                itr_centroids.append({ 'name': cent, 'values': centroids[itr][cent] })
            
            for clust in clusters[itr]:
                itr_clusters.append({ 'name': clust, 'values': clusters[itr][clust] })

            aux_centroids.append(itr_centroids)
            aux_clusters.append(itr_clusters)
        
        rs_centroids.append(aux_centroids)
        rs_clusters.append(aux_clusters)
        
    rs_metrics = []
    for met in mets_results:
        rs_metrics.append({ 'name': met, 'values': mets_results[met] })
    
    response = { 'centroids': rs_centroids, 'clusters': rs_clusters,
                 'results': rs_metrics, }
    return response


def generate_multi_metrics_iterations(dataset_id, algorithm_id, k, n_sim):
    ds = [datasets.load_iris(), datasets.load_wine(),
          datasets.load_diabetes()]

    algorithms = [KMeans, FCMeans]

    metrics = ['inter-cluster', 'cluster-separation', 'abgss',
               'edge-index', 'cluster-connectedness', 'intra-cluster',
               'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
               'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
               'silhouette', 'min-max-cut']

    if dataset_id == 3:
        dataset = pd.read_csv("custom_datasets/dataPhDAlzheimerSemNomes.csv")
        dataset = dataset.iloc[:, [3, 4, 5, 6]].values
        normalize = False
    elif dataset_id == 4:
        dataset = pd.read_csv("custom_datasets/dataPhDAlzheimerSemNomes.csv")
        dataset = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values
        normalize = False
    else:
        ds = [datasets.load_iris(), datasets.load_wine(),
              datasets.load_diabetes()]
        dataset = ds[dataset_id]
        dataset = dataset.data[:, :]
        normalize = True

    algorithm = algorithms[algorithm_id]

    response = execMetrics(
        dataset=dataset, algorithm=algorithm, k=k, metrics=metrics, n_sim=n_sim)
    
    #Write scenarios
    file_name = 'scenarios/ds{}_ag{}_k{}_sim{}.json'.format(dataset_id, algorithm_id, k, n_sim)
    with open(file_name, 'w') as outfile:
        json.dump(response, outfile)

    return response

def prepareToList(centroids, clusters):
    for c in centroids:
        centroids[c]=list(centroids[c])

    for c in clusters:
        for xi in range(len(clusters[c])):
            clusters[c][xi]=list(clusters[c][xi])
    
    return centroids, clusters
