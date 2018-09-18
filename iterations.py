import numpy as np
from algorithms.kmeans import KMeans
from algorithms.fcmeans import FCMeans
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

dataset = [datasets.load_iris(), datasets.load_wine(), datasets.load_diabetes()]

algorithms = [KMeans, FCMeans]

def get_dataset(dataset_id):
    ds = dataset[dataset_id]
    return ds.data[:, :]

def get_algorithm(algorithm_id):
    return algorithms[algorithm_id]

def get_dataset_dimensions(dataset_id):
    ds = dataset[dataset_id]
    return ds.feature_names

def generate_iterations(dataset_id, algorithm_id, k, m):
    X = get_dataset(dataset_id)
    algorithm = get_algorithm(algorithm_id)

    # Normalize the dimension value to a float value with range 0 - 1
    std = MinMaxScaler()
    X = std.fit_transform(X)
    
    if algorithm_id == 1:
        algorithm = algorithm(data=X, k=k, fuzzy_c=m)
    else:
        algorithm = algorithm(data=X, k=k)

    algorithm.fit()
    
    for itr in algorithm.all_centroids:
        for centroid in algorithm.all_centroids[itr]:
            algorithm.all_centroids[itr][centroid] = list(
                algorithm.all_centroids[itr][centroid])
    
    for itr in algorithm.all_clusters:
        for cluster in algorithm.all_clusters[itr]:
            algorithm.all_clusters[itr][cluster] = list(
            algorithm.all_clusters[itr][cluster])
            
            for data in range(len(algorithm.all_clusters[itr][cluster])):
                algorithm.all_clusters[itr][cluster][data] = list(
                    algorithm.all_clusters[itr][cluster][data])
                
    dimensions = get_dataset_dimensions(dataset_id)
    response = { 'centroids': algorithm.all_centroids,
                 'clusters': algorithm.all_clusters,
                 'dimensions': dimensions}
    return response
