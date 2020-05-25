"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run unsupervised pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD neural network model.
"""

import math
import itertools
import random

import hdbscan
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import umap

from bsoid_app.utils.likelihoodprocessing import boxcar_center
from bsoid_app.utils.visuals import *


def bsoid_feats(data: list, fps):
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param data: list of 3D array
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :return f_10fps: 2D array, features
    :return f_10fps_sc: 2D array, standardized/session features
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        logging.info('Extracting features from CSV file {}...'.format(m + 1))
        dataRange = len(data[m])
        dxy_r = []
        dis_r = []
        for r in range(dataRange):
            if r < dataRange - 1:
                dis = []
                for c in range(0, data[m].shape[1], 2):
                    dis.append(np.linalg.norm(data[m][r + 1, c:c + 2] - data[m][r, c:c + 2]))
                dis_r.append(dis)
            dxy = []
            for i, j in itertools.combinations(range(0, data[m].shape[1], 2), 2):
                dxy.append(data[m][r, i:i + 2] - data[m][r, j:j + 2])
            dxy_r.append(dxy)
        dis_r = np.array(dis_r)
        dxy_r = np.array(dxy_r)
        dis_smth = []
        dxy_eu = np.zeros([dataRange, dxy_r.shape[1]])
        ang = np.zeros([dataRange - 1, dxy_r.shape[1]])
        dxy_smth = []
        ang_smth = []
        for l in range(dis_r.shape[1]):
            dis_smth.append(boxcar_center(dis_r[:, l], win_len))
        for k in range(dxy_r.shape[1]):
            for kk in range(dataRange):
                dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                if kk < dataRange - 1:
                    b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                    a_3d = np.hstack([dxy_r[kk, k, :], 0])
                    c = np.cross(b_3d, a_3d)
                    ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                        math.atan2(np.linalg.norm(c),
                                                   np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
            dxy_smth.append(boxcar_center(dxy_eu[:, k], win_len))
            ang_smth.append(boxcar_center(ang[:, k], win_len))
        dis_smth = np.array(dis_smth)
        dxy_smth = np.array(dxy_smth)
        ang_smth = np.array(ang_smth)
        feats.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))
    logging.info('Done extracting features from a total of {} training CSV files.'.format(len(data)))
    for n in range(0, len(feats)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10), len(feats[n][0]), round(fps / 10)):
            if k > round(fps / 10):
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:dxy_smth.shape[0],
                                                             range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                            range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                            range(k - round(fps / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
        logging.info('Done integrating features into 100ms bins from CSV file {}.'.format(n + 1))
        if n > 0:
            f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = np.concatenate((f_10fps_sc, feats1_sc), axis=1)
        else:
            f_10fps = feats1
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = feats1_sc  # scaling is important as I've seen wildly different stdev/feat between sessions
    return f_10fps, f_10fps_sc


def bsoid_umap_embed(f_10fps_sc, umap_params=UMAP_PARAMS):
    """
    Trains UMAP (unsupervised) given a set of features based on (x,y) positions
    :param f_10fps_sc: 2D array, standardized/session features
    :param umap_params: dict, UMAP params in GLOBAL_CONFIG
    :return trained_umap: object, trained UMAP transformer
    :return umap_embeddings: 2D array, embedded UMAP space
    """
    feats_train = f_10fps_sc.T
    logging.info('Transforming all {} instances from {} D into {} D'.format(feats_train.shape[0],
                                                                            feats_train.shape[1],
                                                                            umap_params.get('n_components')))
    trained_umap = umap.UMAP(n_neighbors=int(round(np.sqrt(feats_train.shape[0]))),  # power law
                             **umap_params).fit(feats_train)
    umap_embeddings = trained_umap.embedding_
    logging.info('Done non-linear transformation with UMAP from {} D into {} D.'.format(feats_train.shape[1],
                                                                                        umap_embeddings.shape[1]))
    return trained_umap, umap_embeddings


def bsoid_hdbscan(umap_embeddings, hdbscan_params=HDBSCAN_PARAMS):
    """
    Trains HDBSCAN (unsupervised) given learned UMAP space
    :param umap_embeddings: 2D array, embedded UMAP space
    :param hdbscan_params: dict, HDBSCAN params in GLOBAL_CONFIG
    :return assignments: HDBSCAN assignments
    """
    highest_numulab = -np.infty
    numulab = []
    min_cluster_range = range(6, 21)
    logging.info('Running HDBSCAN on {} instances in {} D space...'.format(*umap_embeddings.shape))
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
                                             min_cluster_size=int(round(0.001 * min_c * umap_embeddings.shape[0])),
                                             **hdbscan_params).fit(umap_embeddings)
        numulab.append(len(np.unique(trained_classifier.labels_)))
        if numulab[-1] > highest_numulab:
            logging.info('Adjusting minimum cluster size to maximize cluster number...')
            highest_numulab = numulab[-1]
            best_clf = trained_classifier
    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)
    logging.info('Done predicting labels for {} instances in {} D space...'.format(*umap_embeddings.shape))
    return assignments, soft_clusters, soft_assignments


def bsoid_nn(feats, labels, hldout=HLDOUT, cv_it=CV_IT, mlp_params=MLP_PARAMS):
    """
    Trains MLP classifier
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, HDBSCAN assignments
    :param hldout: scalar, test partition ratio for validating MLP performance in GLOBAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param mlp_params: dict, MLP parameters in GLOBAL_CONFIG
    :return clf: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    :return nn_assignments: 1D array, neural net predictions
    """
    feats_filt = feats[:, labels >= 0]
    labels_filt = labels[labels >= 0]
    feats_train, feats_test, labels_train, labels_test = train_test_split(feats_filt.T, labels_filt.T,
                                                                          test_size=hldout, random_state=23)
    logging.info(
        'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
            (1 - hldout) * 100))
    classifier = MLPClassifier(**mlp_params)
    classifier.fit(feats_train, labels_train)
    clf = MLPClassifier(**mlp_params)
    clf.fit(feats_filt.T, labels_filt.T)
    nn_assignments = clf.predict(feats.T)
    logging.info('Done training feedforward neural network '
                 'mapping {} features to {} assignments.'.format(feats_train.shape, labels_train.shape))
    scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
    timestr = time.strftime("_%Y%m%d_%H%M")
    if PLOT:
        np.set_printoptions(precision=2)
        titles_options = [("Non-normalized confusion matrix", None),
                          ("Normalized confusion matrix", 'true')]
        titlenames = [("counts"), ("norm")]
        j = 0
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(classifier, feats_test, labels_test,
                                         cmap=plt.cm.Blues,
                                         normalize=normalize)
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            my_file = 'confusion_matrix_{}'.format(titlenames[j])
            disp.figure_.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
            j += 1
        plt.show()
    logging.info(
        'Scored cross-validated feedforward neural network performance.'.format(feats_train.shape, labels_train.shape))
    return clf, scores, nn_assignments


