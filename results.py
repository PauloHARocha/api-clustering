import numpy as np
from kmeans import KMeans
from fcmeans import FCMeans
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

dataset = [datasets.load_iris(), datasets.load_wine(), datasets.load_diabetes()]

algorithms = [KMeans, FCMeans]
# colors = ["green", "red", "yellow", "blue", "purple",
#           "orange", "brown", "pink", "gray", "cyan"] 

def get_dataset(dataset_id):
    ds = dataset[dataset_id]
    return ds.data[:, :]

def get_algorithm(algorithm_id):
    return algorithms[algorithm_id]

def get_dataset_dimensions(dataset_id):
    ds = dataset[dataset_id]
    return ds.feature_names
# def plt_scatter_clusters(clusters, centroids, colors, x, y):
#     for k in clusters:
#         color = colors[k]
#         for xi in clusters[k]:
#             plt.scatter(xi[x], xi[y], color=color, s=10)
#         plt.scatter(centroids[k][x], centroids[k][y],
#                     color="black", marker="x", s=20)
#         plt.ylabel(y)
#         plt.xlabel(x)
#     return plt

def generate_results(dataset_id, algorithm_id, k):
    X = get_dataset(dataset_id)
    algorithm = get_algorithm(algorithm_id)

    # Normalize the dimension value to a float value with range 0 - 1
    std = MinMaxScaler()
    X = std.fit_transform(X)
    
    algorithm = algorithm(data=X, k=k)
    algorithm.fit()
    
    for iter in algorithm.all_centroids:
        for centroid in algorithm.all_centroids[iter]:
            algorithm.all_centroids[iter][centroid] = list(
                algorithm.all_centroids[iter][centroid])
    
    for iter in algorithm.all_clusters:
        for cluster in algorithm.all_clusters[iter]:
            algorithm.all_clusters[iter][cluster] = list(
            algorithm.all_clusters[iter][cluster])
            
            for data in range(len(algorithm.all_clusters[iter][cluster])):
                algorithm.all_clusters[iter][cluster][data] = list(
                    algorithm.all_clusters[iter][cluster][data])
                
    dimensions = get_dataset_dimensions(dataset_id)
    response = { 'centroids': algorithm.all_centroids,
                 'clusters': algorithm.all_clusters,
                 'dimensions': dimensions}
    return response
