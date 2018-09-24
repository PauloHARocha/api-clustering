import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans
from algorithms.metrics import Metrics 

def execMetrics(dataset, algorithm, k, metrics, normalize=True):
    
    if normalize:
        std = MinMaxScaler()
        dataset = std.fit_transform(dataset)
        
    ag_exec = algorithm(data=dataset)
    ag_exec.fit(k=k)

    mets_results = {}

    for met in metrics:
        mets_results[met['name']] = []

    for met in metrics:
        for itr in tqdm(range(len(ag_exec.all_centroids)), desc='{}'.format(met['name'])):
            clusters = ag_exec.all_clusters[itr]
            centroids = ag_exec.all_centroids[itr]
            name = met['name']
            metric = met['metric']

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
    
    rs_centroids=[]
    for itr in ag_exec.all_centroids:
        for centroid in ag_exec.all_centroids[itr]:
            ag_exec.all_centroids[itr][centroid] = list(
                ag_exec.all_centroids[itr][centroid])
        rs_centroids.append(ag_exec.all_centroids[itr])

    rs_clusters=[]
    for itr in ag_exec.all_clusters:
        for cluster in ag_exec.all_clusters[itr]:
            ag_exec.all_clusters[itr][cluster] = list(
                ag_exec.all_clusters[itr][cluster])

            for data in range(len(ag_exec.all_clusters[itr][cluster])):
                ag_exec.all_clusters[itr][cluster][data] = list(
                    ag_exec.all_clusters[itr][cluster][data])
        rs_clusters.append(ag_exec.all_clusters[itr])

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

def generate_metrics_iterations(dataset_id, algorithm_id, k):
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
         },]

    dataset = ds[dataset_id]
    dataset = dataset.data[:, :]

    algorithm = algorithms[algorithm_id]

    met_results = execMetrics(
        dataset=dataset, algorithm=algorithm, k=k, metrics=metrics)
    
    return met_results

# def printMetrics(met_results):
#     subCount = 0
#     for met in met_results:
#         plot_data = []
#         for itr in met_results[met]:
#             plot_data.append(itr)
#         subCount += 1
#         plt.subplot(2,1,subCount)
#         plt.plot(plot_data, '-o')
#         plt.ylabel(met)
#         plt.xlabel('iteration')
#     plt.show()

# def printIterations(centroids, clusters):
#     colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'gray']
#     for itr in clusters:
#         for k in clusters[itr]:
#             color = colors[k]
#             for xi in clusters[itr][k]:
#                 plt.scatter(xi[0], xi[1], color=color, s=10)
#             plt.scatter(centroids[itr][k][0], centroids[itr][k][1],
#                         color="black", marker="x", s=100)
#         plt.figure()
#     plt.show()

# if __name__ == '__main__':
#     datasets = [datasets.load_iris(), datasets.load_wine(),
#                datasets.load_diabetes()]

#     algorithms = [KMeans, FCMeans]

#     metrics = [
#         {'metric': Metrics.silhouette,
#         'name': 'silhouette'
#         },
#         {'metric': Metrics.variance_based_ch,
#         'name': 'ch-index'}]
#         # {'metric': Metrics.gap_statistic,
#         # 'name': 'gap'}]

#     dataset = datasets[0]
#     dataset = dataset.data[:, :]

#     algorithm = algorithms[1]

#     met_results = execMetrics(dataset=dataset, algorithm=algorithm, k=3, metrics=metrics)
    
#     printMetrics(met_results['results'])
#     printIterations(met_results['centroids'], met_results['clusters'])
    
