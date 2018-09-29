def get_parameters():
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
            ],
            'scenarios': {
                'iterations':[
                    {
                        'name': 'Iris/K-Means/k=2/sim=1',
                        'value': 'ds0_ag0_k2_sim1'
                    },
                ],
                'customds':[
                    {
                        'name': 'CustomDs0/FC-means/k=2/sim=30',
                        'value': 'customds0_ag1_k2_sim30'
                    },
                ]
            }
                
                
            
        }
