import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans
from algorithms.metrics import Metrics


def execMetricsDatasets(datasets, algorithm, k, metrics):
    
    rs_metrics = []
    rs_datasets = []
    for met in metrics:
        rs_metrics.append(
            {
                'name': met['name'],
                'values': []
            }
        )
        
    for ds, dataset in enumerate(datasets):
        
        ag_exec = algorithm(data=dataset)
        ag_exec.fit(k=k)

        clusters = ag_exec.clusters
        centroids = ag_exec.centroids

        mets_results = {}

        for met in metrics:
            mets_results[met['name']] = []
            
        for met in tqdm(range(len(metrics)), desc='dataset: {}'.format(ds)):
            name = metrics[met]['name']
            metric = metrics[met]['metric']
            
            if name == 'inter-cluster':
                mets_results[name].append(metric(centroids))
            if name == 'cluster-separation':
                mets_results[name].append(metric(centroids))
            if name == 'intra-cluster':
                mets_results[name].append(metric(dataset, centroids))
            if name == 'ball-hall':
                mets_results[name].append(metric(dataset, centroids))
            elif name == 'ch-index':
                mets_results[name].append(metric(dataset, centroids))
            elif name == 'hartigan':
                mets_results[name].append(metric(dataset, centroids))
            elif name == 'xu-index':
                mets_results[name].append(metric(dataset, centroids))
            elif name == 'wb-index':
                mets_results[name].append(metric(dataset, centroids))
            if name == 'silhouette':
                mets_results[name].append(metric(clusters, len(dataset)))
            elif name == 'gap':
                random_data = np.random.uniform(0, 1, dataset.shape)
                ag_aux = algorithm(data=random_data)
                ag_aux.fit(k=k)
                random_clusters = ag_aux.clusters
                mets_results[name].append(
                    metric(clusters, random_clusters))
        
        for centroid in centroids:
            centroids[centroid] = list(centroids[centroid])
            
        for cluster in clusters:
            clusters[cluster] = list(clusters[cluster])
            for data in range(len(clusters[cluster])):
                clusters[cluster][data] = list(clusters[cluster][data])
        
        rs_centroids = []
        for cent in centroids:
            rs_centroids.append({
                'name': cent,
                'values': centroids[cent]
            })
        rs_clusters = []
        for clust in clusters:
          rs_clusters.append({
              'name': clust,
              'values': clusters[clust]
          }) 

        rs_datasets.append({
            'name': ds,
            'centroids': rs_centroids,
            'clusters': rs_clusters,
            })

        for met in mets_results:
            for m in rs_metrics:
                if met == m['name']:
                    m['values'].append(mets_results[met][0])

    response = {
            'metrics': rs_metrics,
            'datasets': rs_datasets,
        }
    
    return response


def generate_metrics_datasets(algorithm_id, k):
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
         },
        {'metric': Metrics.gap_statistic,
        'name': 'gap'}]

    custom_ds = []
    for idx in range(23): #23
        #Importing dataset
        ds = pd.read_csv("custom_datasets/dataBase_{}.csv".format(idx))
        #Select lines and columns 
        custom_ds.append(ds.iloc[:, [1, 2]].values)
    
    algorithm = algorithms[algorithm_id]

    met_db_results = execMetricsDatasets(
        datasets=custom_ds, algorithm=algorithm, k=k, metrics=metrics)

    return met_db_results

