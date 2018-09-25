import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans
from algorithms.metrics import Metrics


def execMetrics(dataset, algorithm, k, metrics, n_sim, normalize=True):

    if normalize:
        std = MinMaxScaler()
        dataset = std.fit_transform(dataset)

    mets_results = {}
    for met in metrics:
        mets_results[met['name']] = []
    rs_centroids = []
    rs_clusters = []

    ag_exec = algorithm(data=dataset)
    for sim in tqdm(range(n_sim)):    
        ag_exec.fit(k=k)

        aux_results = {}
        for met in metrics:
            aux_results[met['name']] = []
            for itr in tqdm(range(len(ag_exec.all_centroids)), desc='{}'.format(met['name'])):
                clusters = ag_exec.all_clusters[itr]
                centroids = ag_exec.all_centroids[itr]
                name = met['name']
                metric = met['metric']

                if name == 'inter-cluster':
                    aux_results[name].append(metric(centroids))
                if name == 'cluster-separation':
                    aux_results[name].append(metric(centroids))
                if name == 'intra-cluster':
                    aux_results[name].append(metric(dataset, centroids))
                if name == 'ball-hall':
                    aux_results[name].append(metric(dataset, centroids))
                elif name == 'ch-index':
                    aux_results[name].append(metric(dataset, centroids))
                elif name == 'hartigan':
                    aux_results[name].append(metric(dataset, centroids))
                elif name == 'xu-index':
                    aux_results[name].append(metric(dataset, centroids))
                elif name == 'wb-index':
                    aux_results[name].append(metric(dataset, centroids))
                if name == 'silhouette':
                    aux_results[name].append(metric(clusters, len(dataset)))

        for met in metrics:
            mets_results[met['name']].append(aux_results[met['name']])

        centroids = ag_exec.all_centroids
        clusters = ag_exec.all_clusters
        aux_centroids = []
        aux_clusters = []
        for itr in ag_exec.all_centroids:
            itr_centroids = []
            itr_clusters = []
            for centroid in centroids[itr]:
                centroids[itr][centroid] = list(centroids[itr][centroid])
                
            for cluster in clusters[itr]:
                clusters[itr][cluster] = list(clusters[itr][cluster])
                for data in range(len(clusters[itr][cluster])):
                    clusters[itr][cluster][data] = list(
                        clusters[itr][cluster][data])
        
            
            for cent in centroids[itr]:
                itr_centroids.append({
                    'name': cent,
                    'values': centroids[itr][cent]
                })
            
            for clust in clusters[itr]:
                itr_clusters.append({
                    'name': clust,
                    'values': clusters[itr][clust]
                })
            aux_centroids.append(itr_centroids)
            aux_clusters.append(itr_clusters)
        
        rs_centroids.append(aux_centroids)
        rs_clusters.append(aux_clusters)
        
    rs_metrics = []
    for met in mets_results:
        rs_metrics.append(
            {
                'name': met,
                'values': mets_results[met]
            }
        )
    
    response = {
        'centroids': rs_centroids,
        'clusters': rs_clusters,
        'results': rs_metrics,
    }
    return response


def generate_multi_metrics_iterations(dataset_id, algorithm_id, k, n_sim):
    ds = [datasets.load_iris(), datasets.load_wine(),
          datasets.load_diabetes()]

    algorithms = [KMeans, FCMeans]

    metrics = [
        {'metric': Metrics.inter_cluster_statistic,
         'name': 'inter-cluster'
         },
        {'metric': Metrics.cluster_separation,
         'name': 'cluster-separation'
         },
        {'metric': Metrics.intra_cluster_statistic,
         'name': 'intra-cluster'
         },
        {'metric': Metrics.ball_hall_index,
         'name': 'ball-hall'
         },
        {'metric': Metrics.variance_based_ch,
         'name': 'ch-index'
         },
        {'metric': Metrics.hartigan_index,
         'name': 'hartigan'
         },
        {'metric': Metrics.xu_index,
         'name': 'xu-index'
         },
        {'metric': Metrics.wb_index,
         'name': 'wb-index'
         },
        {'metric': Metrics.silhouette,
         'name': 'silhouette'
         }, ]

    dataset = ds[dataset_id]
    dataset = dataset.data[:, :]

    algorithm = algorithms[algorithm_id]

    met_results = execMetrics(
        dataset=dataset, algorithm=algorithm, k=k, metrics=metrics, n_sim=n_sim)

    return met_results


