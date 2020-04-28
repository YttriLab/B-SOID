################### THINGS YOU PROBABLY DONT'T WANT TO CHANGE ###################

import logging
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level='INFO',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout)

# EM_GMM parameters
EMGMM_PARAMS = {
    'n_components': 30,
    'covariance_type': 'full',  # t-sne structure means nothing.
    'tol': 0.001,
    'reg_covar': 1e-06,
    'max_iter': 100,
    'n_init': 20,  # 20 iterations to escape poor initialization
    'init_params': 'random',  # random initialization
    'random_state': 23,
    'verbose': 1  # set this to 0 if you don't want to show progress for em-gmm.
}

# Multi-class support vector machine classifier params
SVM_PARAMS = {
    'C': 10,  # 100 units, 10 layers
    'gamma': 0.5,  # logistics appears to outperform tanh and relu
    'probability': True,
    'random_state': 0,  # adaptive or constant, not too much of a diff
    'verbose': 0  # set to 1 for tuning your feedforward neural network
}

HLDOUT = 0.2  # Test partition ratio to validate clustering separation.
CV_IT = 10  # Number of iterations for cross-validation to show it's not over-fitting.
