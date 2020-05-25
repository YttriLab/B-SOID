################### THINGS YOU PROBABLY DONT'T WANT TO CHANGE ###################

import logging
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level='INFO',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout)

# UMAP params, nonlinear transform
UMAP_PARAMS = {
    'n_components': 3,
    'min_dist': 0.0,  # small value
    'random_state': 23,
}

# HDBSCAN params, density based clustering
HDBSCAN_PARAMS = {
    'min_samples': 10  # small value
}

# Feedforward neural network (MLP) params
MLP_PARAMS = {
    'hidden_layer_sizes': (100, 10),  # 100 units, 10 layers
    'activation': 'logistic',  # logistics appears to outperform tanh and relu
    'solver': 'adam',
    'learning_rate': 'constant',
    'learning_rate_init': 0.001,  # learning rate not too high
    'alpha': 0.0001,  # regularization default is better than higher values.
    'max_iter': 1000,
    'early_stopping': False,
    'verbose': 0  # set to 1 for tuning your feedforward neural network
}

HLDOUT = 0.2  # Test partition ratio to validate clustering separation.
CV_IT = 10  # Number of iterations for cross-validation to show it's not over-fitting.
