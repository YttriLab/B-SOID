"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD behavioral model.
"""

import math

import numpy as np
from sklearn import mixture
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from bsoid_py.utils.likelihoodprocessing import boxcar_center
from bsoid_py.utils.visuals import *


def bsoid_tsne(data: list, bodyparts=BODYPARTS, fps=FPS, comp=COMP, tsne_params=TSNE_PARAMS):
    """
    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions
    :param data: list of 3D array
    :param bodyparts: dict, body parts with their orders in LOCAL_CONFIG
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :return f_10fps: 2D array, features
    :retrun f_10fps_sc: 2D array, standardized features
    :return trained_tsne: 2D array, trained t-SNE space
    """
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    feats = []
    for m in range(len(data)):
        logging.info('Extracting features from CSV file {}...'.format(m + 1))
        dataRange = len(data[m])
        fpd = data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1'):2 * bodyparts.get('Forepaw/Shoulder1') + 2] - \
              data[m][:, 2 * bodyparts.get('Forepaw/Shoulder2'):2 * bodyparts.get('Forepaw/Shoulder2') + 2]
        cfp = np.vstack(((data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1')] +
                          data[m][:, 2 * bodyparts.get('Forepaw/Shoulder2')]) / 2,
                         (data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1') + 1] +
                          data[m][:, 2 * bodyparts.get('Forepaw/Shoulder1') + 1]) / 2)).T
        cfp_pt = np.vstack(([cfp[:, 0] - data[m][:, 2 * bodyparts.get('Tailbase')],
                             cfp[:, 1] - data[m][:, 2 * bodyparts.get('Tailbase') + 1]])).T
        chp = np.vstack((((data[m][:, 2 * bodyparts.get('Hindpaw/Hip1')] +
                           data[m][:, 2 * bodyparts.get('Hindpaw/Hip2')]) / 2),
                         ((data[m][:, 2 * bodyparts.get('Hindpaw/Hip1') + 1] +
                           data[m][:, 2 * bodyparts.get('Hindpaw/Hip2') + 1]) / 2))).T
        chp_pt = np.vstack(([chp[:, 0] - data[m][:, 2 * bodyparts.get('Tailbase')],
                             chp[:, 1] - data[m][:, 2 * bodyparts.get('Tailbase') + 1]])).T
        sn_pt = np.vstack(([data[m][:, 2 * bodyparts.get('Snout/Head')] - data[m][:, 2 * bodyparts.get('Tailbase')],
                            data[m][:, 2 * bodyparts.get('Snout/Head') + 1] - data[m][:,
                                                                              2 * bodyparts.get('Tailbase') + 1]])).T
        fpd_norm = np.zeros(dataRange)
        cfp_pt_norm = np.zeros(dataRange)
        chp_pt_norm = np.zeros(dataRange)
        sn_pt_norm = np.zeros(dataRange)
        for i in range(1, dataRange):
            fpd_norm[i] = np.array(np.linalg.norm(fpd[i, :]))
            cfp_pt_norm[i] = np.linalg.norm(cfp_pt[i, :])
            chp_pt_norm[i] = np.linalg.norm(chp_pt[i, :])
            sn_pt_norm[i] = np.linalg.norm(sn_pt[i, :])
        fpd_norm_smth = boxcar_center(fpd_norm, win_len)
        sn_cfp_norm_smth = boxcar_center(sn_pt_norm - cfp_pt_norm, win_len)
        sn_chp_norm_smth = boxcar_center(sn_pt_norm - chp_pt_norm, win_len)
        sn_pt_norm_smth = boxcar_center(sn_pt_norm, win_len)
        sn_pt_ang = np.zeros(dataRange - 1)
        sn_disp = np.zeros(dataRange - 1)
        pt_disp = np.zeros(dataRange - 1)
        for k in range(0, dataRange - 1):
            b_3d = np.hstack([sn_pt[k + 1, :], 0])
            a_3d = np.hstack([sn_pt[k, :], 0])
            c = np.cross(b_3d, a_3d)
            sn_pt_ang[k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                  math.atan2(np.linalg.norm(c), np.dot(sn_pt[k, :], sn_pt[k + 1, :])))
            sn_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1] -
                data[m][k, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1])
            pt_disp[k] = np.linalg.norm(
                data[m][k + 1, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1] -
                data[m][k, 2 * bodyparts.get('Tailbase'):2 * bodyparts.get('Tailbase') + 1])
        sn_pt_ang_smth = boxcar_center(sn_pt_ang, win_len)
        sn_disp_smth = boxcar_center(sn_disp, win_len)
        pt_disp_smth = boxcar_center(pt_disp, win_len)
        feats.append(np.vstack((sn_cfp_norm_smth[1:], sn_chp_norm_smth[1:], fpd_norm_smth[1:],
                                sn_pt_norm_smth[1:], sn_pt_ang_smth[:], sn_disp_smth[:], pt_disp_smth[:])))
    logging.info('Done extracting features from a total of {} training CSV files.'.format(len(data)))
    if comp == 0:
        f_10fps = []
        f_10fps_sc = []
        trained_tsne = []
    for n in range(0, len(feats)):
        feats1 = np.zeros(len(data[n]))
        for k in range(round(fps / 10) - 1, len(feats[n][0]), round(fps / 10)):
            if k > round(fps / 10) - 1:
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                                    np.sum((feats[n][4:7, range(k - round(fps / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:4, range(k - round(fps / 10), k)]), axis=1),
                                    np.sum((feats[n][4:7, range(k - round(fps / 10), k)]), axis=1))).reshape(
                    len(feats[0]), 1)
        logging.info('Done integrating features into 100ms bins from CSV file {}.'.format(n + 1))
        if comp == 1:
            if n > 0:
                f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            else:
                f_10fps = feats1
        else:
            f_10fps.append(feats1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_stnd = scaler.transform(feats1.T).T
            f_10fps_sc.append(feats1_stnd)
            if f_10fps_sc[n].shape[1] < 10000:
                print("Insufficient data, exiting...")
                exit()
            np.random.seed(23)  # For reproducibility
            logging.info('Training t-SNE to embed {} instances from {} D '
                         'into 3 D from CSV file {}...'.format(f_10fps_sc[n].shape[1], f_10fps_sc[n].shape[0],
                                                               n + 1))
            trained_tsne_i = TSNE(perplexity=(f_10fps_sc[n].shape[1] / 300),
                                  early_exaggeration=(f_10fps_sc[n].shape[1] / 2500),
                                  learning_rate=2500,
                                  **tsne_params).fit_transform(f_10fps_sc[n].T)
            trained_tsne.append(trained_tsne_i)
            logging.info('Done embedding into 3 D.')
    if comp == 1:
        if f_10fps.shape[1] < 10000:
            print("Insufficient data, exiting...")
            exit()
        scaler = StandardScaler()
        scaler.fit(f_10fps.T)
        f_10fps_sc = scaler.transform(f_10fps.T).T
        np.random.seed(23)  # For reproducibility
        logging.info('Training t-SNE to embed {} instances from {} D '
                     'into 3 D from a total of {} CSV files...'.format(f_10fps_sc.shape[1], f_10fps_sc.shape[0],
                                                                       len(data)))
        trained_tsne = TSNE(perplexity=(f_10fps_sc.shape[1] / 300),  # 1% data
                            early_exaggeration=(f_10fps_sc.shape[1] / 2500),  # scale exag by data
                            learning_rate=2500,  # n/exag
                            **tsne_params).fit_transform(f_10fps_sc.T)
        logging.info('Done embedding into 3 D.')
    return f_10fps, f_10fps_sc, trained_tsne


def bsoid_gmm(trained_tsne, comp=COMP, emgmm_params=EMGMM_PARAMS):
    """
    Trains EM-GMM (unsupervised) given learned t-SNE space
    :param trained_tsne: 2D array, trained t-SNE space
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :param emgmm_params: dict, EMGMM_PARAMS in GLOBAL_CONFIG
    :return assignments: Converged EM-GMM group assignments
    """
    if comp == 1:
        logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne.shape))
        gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne)
        logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
        assigns = gmm.predict(trained_tsne)
    else:
        assigns = []
        for i in tqdm(range(len(trained_tsne))):
            logging.info('Running EM-GMM on {} instances in {} D space...'.format(*trained_tsne[i].shape))
            gmm = mixture.GaussianMixture(**emgmm_params).fit(trained_tsne[i])
            logging.info('Predicting labels for {} instances in {} D space...'.format(*trained_tsne[i].shape))
            assign = gmm.predict(trained_tsne[i])
            assigns.append(assign)
    logging.info('Done predicting labels for {} instances in {} D space...'.format(*trained_tsne.shape))
    uk = list(np.unique(assigns))
    assignments_li = []
    for i in assigns:
        indexVal = uk.index(i)
        assignments_li.append(indexVal)
    assignments = np.array(assignments_li)
    return assignments


def bsoid_nn(feats, labels, comp=COMP, hldout=HLDOUT, cv_it=CV_IT, mlp_params=MLP_PARAMS):
    """
    Trains MLP classifier
    :param feats: 2D array, original feature space, standardized
    :param labels: 1D array, GMM output assignments
    :param hldout: scalar, test partition ratio for validating MLP performance in GLOBAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in GLOBAL_CONFIG
    :param mlp_params: dict, MLP parameters in GLOBAL_CONFIG
    :return classifier: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    """
    if comp == 1:
        feats_train, feats_test, labels_train, labels_test = train_test_split(feats.T, labels.T, test_size=hldout,
                                                                              random_state=23)
        logging.info(
            'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
                (1 - hldout) * 100))
        classifier = MLPClassifier(**mlp_params)
        classifier.fit(feats_train, labels_train)
        logging.info('Done training feedforward neural network mapping {} features to {} assignments.'.format(
            feats_train.shape, labels_train.shape))
        logging.info('Predicting randomly sampled (non-overlapped) assignments '
                     'using the remaining {}%...'.format(HLDOUT * 100))
        scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
        timestr = time.strftime("_%Y%m%d_%H%M")
        if PLOT_TRAINING:
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
                my_file = 'confusion_matrix_{}_'.format(titlenames[j])
                disp.figure_.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
                j += 1
            plt.show()
    else:
        classifier = []
        scores = []
        for i in range(len(feats)):
            feats_train, feats_test, labels_train, labels_test = train_test_split(feats[i].T, labels[i].T,
                                                                                  test_size=hldout,
                                                                                  random_state=23)
            logging.info(
                'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
                    (1 - hldout) * 100))
            clf = MLPClassifier(**mlp_params)
            clf.fit(feats_train, labels_train)
            classifier.append(clf)
            logging.info(
                'Done training feedforward neural network mapping {} features to {} assignments.'.format(
                    feats_train.shape, labels_train.shape))
            logging.info('Predicting randomly sampled (non-overlapped) assignments '
                         'using the remaining {}%...'.format(HLDOUT * 100))
            sc = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
            timestr = time.strftime("_%Y%m%d_%H%M")
            if PLOT_TRAINING:
                np.set_printoptions(precision=2)
                titles_options = [("Non-normalized confusion matrix", None),
                                  ("Normalized confusion matrix", 'true')]
                j = 0
                titlenames = [("counts"), ("norm")]
                for title, normalize in titles_options:
                    disp = plot_confusion_matrix(classifier, feats_test, labels_test,
                                                 cmap=plt.cm.Blues,
                                                 normalize=normalize)
                    disp.ax_.set_title(title)
                    print(title)
                    print(disp.confusion_matrix)
                    my_file = 'confusion_matrix_clf{}_{}_'.format(i, titlenames[j])
                    disp.figure_.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
                    j += 1
                plt.show()
    logging.info(
        'Scored cross-validated feedforward neural network performance.'.format(feats_train.shape, labels_train.shape))
    return classifier, scores


def main(train_folders: list):
    """
    :param train_folders: list, training data folders
    :return f_10fps: 2D array, features
    :return trained_tsne: 2D array, trained t-SNE space
    :return gmm_assignments: Converged EM-GMM group assignments
    :return classifier: obj, MLP classifier
    :return scores: 1D array, cross-validated accuracy
    """
    import bsoid_py.utils.likelihoodprocessing
    filenames, training_data, perc_rect = bsoid_py.utils.likelihoodprocessing.main(train_folders)
    f_10fps, f_10fps_sc, trained_tsne = bsoid_tsne(training_data)
    gmm_assignments = bsoid_gmm(trained_tsne)
    classifier, scores = bsoid_nn(f_10fps, gmm_assignments)
    if PLOT_TRAINING:
        plot_classes(trained_tsne, gmm_assignments)
        plot_accuracy(scores)
        plot_feats(f_10fps, gmm_assignments)
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores
