from flask import Flask
from flask_restful import Resource, Api
from iterations import generate_iterations
from comparison import generate_comparision
from flask_cors import CORS

app = Flask(__name__)

api = Api(app)
CORS(app)

class Iterations(Resource):
    def get(self, dataset_id, algorithm_id, k):
        return generate_iterations(dataset_id, algorithm_id, k)

class Comparision(Resource):
    def get(self, dataset_id, algorithm_id, k, k_min, k_max, n_sim):
        return generate_comparision(dataset_id, algorithm_id, k, k_min, k_max, n_sim)

api.add_resource(Iterations, '/iterations/<int:dataset_id>/<int:algorithm_id>/<int:k>')

api.add_resource(
    Comparision, '/comparision/<int:dataset_id>/<int:algorithm_id>/<int:k>/<int:k_min>/<int:k_max>/<int:n_sim>')

if __name__ == '__main__':
    app.run(debug=True)
