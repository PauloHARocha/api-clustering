import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from algorithms.metrics import Metrics
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans

dataset = [datasets.load_iris(), datasets.load_wine(),
           datasets.load_diabetes()]

algorithms = [KMeans, FCMeans]


def get_dataset(dataset_id):
    ds = dataset[dataset_id]
    return ds.data[:, :]


def get_algorithm(algorithm_id):
    return algorithms[algorithm_id]


def get_dataset_dimensions(dataset_id):
    ds = dataset[dataset_id]
    return ds.feature_names


def generate_comparision(dataset_id, algorithm_id, k, k_min, k_max, n_sim):
    X = get_dataset(dataset_id)

    # Normalize the dimension value to a float value with range 0 - 1
    std = MinMaxScaler()
    X = std.fit_transform(X)

    response = {
        'k_min': k_min,
        'k_max': k_max,
        'n_sim': n_sim
        }
    return generate_metrics_results(X, algorithm_id, response)

def generate_metrics_results(X, algorithm_id, response):
    metrics = ['silhouete', 'calinskiHarabaszIndex', 'gap'] 
    
    algorithm = get_algorithm(0)
    algorithm = algorithm(data=X)
    name = algorithm.__str__()

    response[name] = {}

    rng = range(response['k_min'], response['k_max'] + 1)  # 2-16

    for k in rng:
        response[name][k] = {}
        for met in metrics:
            response[name][k][met] = {}
        for n in range(response['n_sim']):
                    
            algorithm.fit(k=k)
            
            for met in metrics:
                if met == 'silhouete':
                    response[name][k][met][n] = Metrics.silhouette(
                        algorithm.clusters, len(X))
                elif met == 'calinskiHarabaszIndex':
                    response[name][k][met][n] = Metrics.variance_based_ch(
                        X, algorithm.centroids)
                elif met == 'gap':
                    clusters = algorithm.clusters

                    random_data = np.random.uniform(0, 1, X.shape)
                    algorithm.fit(data=random_data)
                    random_clusters = algorithm.clusters

                    response[name][k][met][n] = Metrics.gap_statistic(
                        clusters, random_clusters)

        for met in metrics:
            met_mean_std = []
            for n in response[name][k][met]:
                met_mean_std.append(response[name][k][met][n])
            response[name][k][met]['mean'] = np.mean(met_mean_std)
            response[name][k][met]['std'] = np.std(met_mean_std)
            
    
    return response

