import numpy as np

rf_hyperparameter_space = {'min_samples_split':(5,25), 
                    'min_samples_leaf':(10,100), 
                    'n_estimators':(100,500),
                    'max_depth':(5,50), 
                    'max_features':(0.1,0.5),
                    'random_state':[42]}

sgd_hyperparameter_space = {
                'sgd__alpha':(10**-7, 10**-1),
                'sgd__l1_ratio':(0.01,1.0),
                'sgd__max_iter':[100,1000] 
}

# FOR SKOPT/BAYESSEARCH
lr_hyperparameter_space = {
                'lr__l1_ratio':(0.0001, 1.0),
                'lr__C':(0.00001, 100.0)
}

# FOR GRIDSEARCH
# lr_hyperparameter_space = {
#                 'lr__l1_ratio':[0.0001, 0.001, 0.01, 0.1, 1.0],
#                 'lr__C':[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0,100.0]
#                 #'lr__solver':['liblinear', 'saga', 'lbfgs'],
#                 #'lr__tol':[1E-5, 1E-4, 1E-3, 1E-2]
# }


hyper_dict = {'rf_hyperparameter_space':rf_hyperparameter_space,
            'lr_hyperparameter_space':lr_hyperparameter_space,
            'sgd_hyperparameter_space':sgd_hyperparameter_space}