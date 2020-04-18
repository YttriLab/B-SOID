import numpy as np
import math
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
    'covariance_type': 'full',
    'tol': 0.001,
    'reg_covar': 1e-06,
    'max_iter': 100,
    'n_init': 20, # 30 iterations to escape poor initialization
    'init_params': 'random', # random initialization
    'random_state': 23,
    'verbose': 1 # set this to 0 if you don't want to show progress for em-gmm.
}

#TODO figure out why Bayesian inference isn't converging for bsoid

# EMGMM_PARAMS = {
#     'n_components': 50,  # initialize k classes
#     'covariance_type': 'full',
#     'tol': 0.001,
#     'reg_covar': 1e-06,
#     'max_iter': 100,
#     'n_init': 30, # 30 iterations to escape poor initialization
#     'init_params': 'random', # random initialization
#     'weight_concentration_prior_type': 'dirichlet_process',
#     'random_state': 23,
#     'verbose': 1, # set this to 0 if you don't want to show progress for em-gmm.
#     'verbose_interval': 50
# }


# SVM parameters
SVM_PARAMS = {
    'gamma': 0.5,  # Kernel coefficient
    'C': 10,  # Regularization parameter
    'probability': True,
    'random_state': 23,
    'verbose': 0 # set this to 1 if you want to show optimization progress
}

HLDOUT = 0.2  # Test partition ratio to validate clustering separation.
CV_IT = 20  # Number of iterations for cross-validation to show it's not over-fitting.

# IF YOU'D LIKE TO SKIP PLOTTING/CREATION OF VIDEOS, change below plot settings to False
PLOT_TRAINING = True
GEN_VIDEOS = True