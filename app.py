from flask import Flask
from flask_restful import Resource, Api
from iterations import generate_iterations
from comparison import generate_comparision
from metricsIterations import generate_metrics_iterations
from multiMetricsIterations import generate_multi_metrics_iterations
from metricsCustomDB import generate_metrics_datasets
from flask_cors import CORS

app = Flask(__name__)

api = Api(app)
CORS(app)

class Param(Resource):
    def get(self):
        return {
                'datasets': [
                    {
                        'name': 'Íris',
                        'value': 0
                    },
                    {
                        'name': 'Wine',
                        'value': 1
                    },
                    {
                        'name': 'Diabetes',
                        'value': 2
                    },
                ],
                'algorithms': [
                    {
                        'name': 'K-Means',
                        'value': 0
                    },
                    {
                        'name': 'FC-Means',
                        'value': 1
                    },
                ]
            }

class Iterations(Resource):
    def get(self, dataset_id, algorithm_id, k, m):
        return generate_iterations(dataset_id, algorithm_id, k, m)

class Comparision(Resource):
    def get(self, dataset_id, algorithm_id, k_min, k_max, n_sim):
        return generate_comparision(dataset_id, algorithm_id, k_min, k_max, n_sim)

class MetricIterations(Resource):
    def get(self, dataset_id, algorithm_id, k):
        return generate_metrics_iterations(dataset_id, algorithm_id, k)

class MultiMetricIterations(Resource):
    def get(self, dataset_id, algorithm_id, k, n_sim):
        return generate_multi_metrics_iterations(dataset_id, algorithm_id, k, n_sim)

class MetricCustomDS(Resource):
    def get(self, ds_idx, algorithm_id, k):
        return generate_metrics_datasets(algorithm_id, k, ds_idx)

api.add_resource(Param, '/param')

api.add_resource(Iterations, '/iterations/<int:dataset_id>/<int:algorithm_id>/<int:k>/<int:m>')

api.add_resource(
    Comparision, '/comparision/<int:dataset_id>/<int:algorithm_id>/<int:k_min>/<int:k_max>/<int:n_sim>')

api.add_resource(
    MetricIterations, '/metrics_iterations/<int:dataset_id>/<int:algorithm_id>/<int:k>')

api.add_resource(
    MultiMetricIterations, '/multi_metrics_iterations/<int:dataset_id>/<int:algorithm_id>/<int:k>/<int:n_sim>')

api.add_resource(
    MetricCustomDS, '/metrics_customds/<int:ds_idx>/<int:algorithm_id>/<int:k>')

if __name__ == '__main__':
    app.run(debug=True)
