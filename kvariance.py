import numpy as np
import json
from tqdm import tqdm
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from algorithms.metrics import Metrics
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans

def generate_kvariance(dataset_id, algorithm_id, k_min, k_max, n_sim):
    ds = [datasets.load_iris(), datasets.load_wine(),
          datasets.load_diabetes()]

    algorithms = [KMeans, FCMeans]

    metrics = ['inter-cluster', 'cluster-separation', 'abgss',
               'edge-index', 'cluster-connectedness', 'intra-cluster',
               'ball-hall', 'intracluster-entropy', 'ch-index', 'hartigan',
               'xu-index', 'wb-index', 'dunn-index', 'davies-bouldin', 'cs-measure',
               'silhouette', 'min-max-cut', 'gap']

    dataset = ds[dataset_id]
    dataset = dataset.data[:, :]

    algorithm = algorithms[algorithm_id]
    k_rng = range(k_min, k_max+1)
    response = execMetrics(
        dataset=dataset, algorithm=algorithm, k_rng=k_rng, metrics=metrics, n_sim=n_sim)

    #Write scenarios
    file_name = 'scenarios/ds{}_ag{}_k{}-{}_sim{}.json'.format(
        dataset_id, algorithm_id, k_min, k_max, n_sim)
    with open(file_name, 'w') as outfile:
        json.dump(response, outfile)

    return response


def execMetrics(dataset, algorithm, k_rng, metrics, n_sim, normalize=True):

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

        for met in metrics:
            mets_results[met].append(aux_metrics[met])

    rs_metrics = []
    for met in mets_results:
        rs_metrics.append({'name': met, 'values': mets_results[met]})

    response = {'centroids': rs_centroids, 'clusters': rs_clusters,
                'metrics': rs_metrics, }
    return response

