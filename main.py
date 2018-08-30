from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class Results(Resource):
    def get(self, dataset_id, algorithm_id, k_id):
        return {'Dataset': dataset_id,
                'Algorithm': algorithm_id,
                'K': k_id}

api.add_resource(Results, '/<dataset_id>/<algorithm_id>/<k_id>')

if __name__ == '__main__':
    app.run(debug=True)
