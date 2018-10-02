from sklearn import datasets
def get_parameters():
    return {
            'datasets': [
                {
                    'name': 'Íris',
                    'value': 0,
                    'dimensions': datasets.load_iris()['feature_names']
                },
                {
                    'name': 'Wine',
                    'value': 1,
                    'dimensions': datasets.load_wine()['feature_names']
                },
                {
                    'name': 'Diabetes',
                    'value': 2,
                    'dimensions': datasets.load_diabetes()['feature_names']
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
            ],
            'scenarios': {
                'iterations':[
                    {
                        'name': 'Iris - FC-Means - k=2 - n_sim=30',
                        'value': 'ds0_ag1_k2_sim30'
                    },
                    {
                        'name': 'Iris - FC-Means - k=3 - n_sim=30',
                        'value': 'ds0_ag1_k3_sim30'
                    },
                    {
                        'name': 'Iris - FC-Means - k=4 - n_sim=30',
                        'value': 'ds0_ag1_k4_sim30'
                    },
                    {
                        'name': 'Wine - FC-Means - k=2 - n_sim=30',
                        'value': 'ds1_ag1_k2_sim30'
                    },
                    {
                        'name': 'Wine - FC-Means - k=3 - n_sim=30',
                        'value': 'ds1_ag1_k3_sim30'
                    },
                    {
                        'name': 'Wine - FC-Means - k=4 - n_sim=30',
                        'value': 'ds1_ag1_k4_sim30'
                    },
                ],
                'customds':[
                    {
                        'name': 'CustomDs=0 - FC-means - k=2 - n_sim=30',
                        'value': 'customds0_ag1_k2_sim30'
                    },
                    {
                        'name': 'CustomDs=0 - FC-means - k=3 - n_sim=30',
                        'value': 'customds0_ag1_k3_sim30'
                    },
                    {
                        'name': 'CustomDs=0 - FC-means - k=4 - n_sim=30',
                        'value': 'customds0_ag1_k4_sim30'
                    },
                    {
                        'name': 'CustomDs=0 - FC-means - k=5 - n_sim=30',
                        'value': 'customds0_ag1_k5_sim30'
                    },
                    {
                        'name': 'CustomDs=1 - FC-means - k=2 - n_sim=30',
                        'value': 'customds1_ag1_k2_sim30'
                    },
                    {
                        'name': 'CustomDs=1 - FC-means - k=3 - n_sim=30',
                        'value': 'customds1_ag1_k3_sim30'
                    },
                    {
                        'name': 'CustomDs=1 - FC-means - k=4 - n_sim=30',
                        'value': 'customds1_ag1_k4_sim30'
                    },
                    {
                        'name': 'CustomDs=1 - FC-means - k=5 - n_sim=30',
                        'value': 'customds1_ag1_k5_sim30'
                    },
                ]
            }
                
                
            
        }
