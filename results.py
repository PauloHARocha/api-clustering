import matplotlib.pyplot as plt
import pandas as pd
from kmeans import KMeans
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

dataset = [datasets.load_iris()]
algorithms = [KMeans]
colors = ["green", "red", "yellow", "blue", "purple",
          "orange", "brown", "pink", "gray", "cyan"]
image_folder = 'images/' 

def get_dataset(dataset_id):
    ds = dataset[dataset_id]
    return ds.data[:, :]

def get_algorithm(algorithm_id):
    return algorithms[algorithm_id]

def plt_scatter_clusters(clusters, centroids, colors, x, y):
    for k in clusters:
        color = colors[k]
        for xi in clusters[k]:
            plt.scatter(xi[x], xi[y], color=color, s=10)
        plt.scatter(centroids[k][x], centroids[k][y],
                    color="black", marker="x", s=100)
        plt.ylabel(y)
        plt.xlabel(x)
    return plt

def generate_results(dataset_id, algorithm_id, k):
    X = get_dataset(dataset_id)
    algorithm = get_algorithm(algorithm_id)

    # Normalize the dimension value to a float value with range 0 - 1
    std = MinMaxScaler()
    X = std.fit_transform(X)
    
    image_paths = {}
    image_id = 0
    algorithm = algorithm(data=X, k=k)
    algorithm.fit()
        
    for x in range(0, 4):
        for y in range(x, 4):
            if x != y:
                plt.figure()
                plt_scatter_clusters(
                    algorithm.clusters, algorithm.centroids, colors, x, y
                    )
                plt.savefig('{}{}-{}'.format(image_folder, str(x), str(y)))
                image_paths[image_id] = '{}{}-{}'.format(image_folder, str(x), str(y))
                image_id += 1
    return image_paths
