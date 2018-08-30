from flask import Flask
from flask_restful import Resource, Api
from results import generate_results

app = Flask(__name__)

api = Api(app)

class Results(Resource):
    def get(self, dataset_id, algorithm_id, k):
        return generate_results(dataset_id, algorithm_id, k)

api.add_resource(Results, '/<int:dataset_id>/<int:algorithm_id>/<int:k>')

if __name__ == '__main__':
    app.run(debug=True)
