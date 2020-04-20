################### THINGS YOU PROBABLY DONT'T WANT TO CHANGE ###################

import logging
import sys

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level='INFO',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout)

# TSNE parameters, can tweak if you are getting undersplit/oversplit behaviors
# the missing perplexity is scaled with data size (1% of data for nearest neighbors)
TSNE_PARAMS = {
    'n_components': 3, # 3 is good, 2 will not create unique pockets, 4 will screw GMM up (curse of dimensionality)
    'learning_rate': 1000,
    'n_jobs': -1, # all cores being used
    'verbose': 2 # check points being told for users to see convergence, and total error/50iterations
}

# EM_GMM parameters
EMGMM_PARAMS = {
    'n_components': 30,
    'covariance_type': 'full', # t-SNE shape means nothing, so full covariance.
    'tol': 0.001,
    'reg_covar': 1e-06,
    'max_iter': 100,
    'n_init': 10, # 30 iterations to escape poor initialization
    'init_params': 'random', # random initialization
    'random_state': 23,
    'verbose': 1 # set this to 0 if you don't want to show progress for em-gmm.
}

# SVM parameters
SVM_PARAMS = {
    'gamma': 0.5,  # Kernel coefficient
    'C': 10,  # Regularization parameter
    'probability': True,
    'random_state': 23,
    'verbose': 0 # set this to 1 if you want to show optimization progress
}

HLDOUT = 0.2  # Test partition ratio to validate clustering separation.
CV_IT = 10  # Number of iterations for cross-validation to show it's not over-fitting.
