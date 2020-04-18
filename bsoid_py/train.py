"""
Based on the natural statistics of the mouse configuration using (x,y) positions,
we distill information down to 3 dimensions and run pattern recognition.
Then, we utilize these output and original feature space to train a B-SOiD behavioral model.
"""

from bsoid_py.utils.visuals import *
from bsoid_py.utils.likelihoodprocessing import boxcar_center
from sklearn.manifold import TSNE
from sklearn import mixture, svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def bsoid_tsne(data, bodyparts=BODYPARTS, fps=FPS, comp=COMP):
    """
    Trains t-SNE (unsupervised) given a set of features based on (x,y) positions
    :param data: list of 3D array
    :param bodyparts: dict, body parts with their orders in LOCAL_CONFIG
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param comp: boolean (0 or 1), argument to compile data or not in LOCAL_CONFIG
    :return f_10fps: 2D array, features
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
        f_10fps_stnd = []
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
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_stnd = scaler.transform(feats1.T)
            f_10fps.append(feats1)
            f_10fps_stnd.append(feats1_stnd)
            if f_10fps[n].shape[1] < 3000:
                print("Insufficient data, exiting...")
                exit()
            else:
                p = round(f_10fps[n].shape[1] / 60)  # 3/60 = 5% of data for community.
                lr = round(f_10fps[n].shape[1] / 12)  # gradient descent learning rate 500-1000 for larger data.
            np.random.seed(23)  # For reproducibility
            logging.info('Training t-SNE to embed {} instances from {} D '
                         'into 3 D from CSV file {}...'.format(f_10fps[n].shape[1], f_10fps.shape[0],
                                                               n + 1))
            trained_tsne_i = TSNE(n_components=3, perplexity=p, early_exaggeration=12, learning_rate=lr,
                                  n_jobs=-1, verbose=2).fit_transform(f_10fps_stnd[n].T)
            trained_tsne.append(trained_tsne_i)
            logging.info('Done embedding into 3 D.')
    if comp == 1:
        if f_10fps.shape[1] < 3000:
            print("Insufficient data, exiting...")
            exit()
        else:
            p = round(f_10fps.shape[1] / 60)  # 3/60 = 5% of data for community.
            lr = round(f_10fps.shape[1] / 12)  # gradient descent learning rate 500-1000 for larger data.
        scaler = StandardScaler()
        scaler.fit(f_10fps.T)
        f_10fps_stnd = scaler.transform(f_10fps.T)
        np.random.seed(23)  # For reproducibility
        logging.info('Training t-SNE to embed {} instances from {} D '
                     'into 3 D from a total of {} CSV files...'.format(f_10fps.shape[1], f_10fps.shape[0],
                                                                       len(data)))
        trained_tsne = TSNE(n_components=3, perplexity=p, early_exaggeration=12, learning_rate=lr,
                            n_jobs=-1, verbose=2).fit_transform(f_10fps_stnd)
        logging.info('Done embedding into 3 D.')
    return f_10fps, trained_tsne


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


def bsoid_svm(feats, labels, hldout=HLDOUT, cv_it=CV_IT, svm_params=SVM_PARAMS):
    """
    :param feats: 2D array, original feature space
    :param labels: 1D array, GMM output assignments
    :param hldout: scalar, test partition ratio for validating SVM performance in LOCAL_CONFIG
    :param cv_it: scalar, iterations for cross-validation in LOCAL_CONFIG
    :param svm_params: dict, SVM parameters in GLOBAL_CONFIG
    :return classifier: obj, SVM.sklearn.svm._classes.SVC classifier
    :return scores: 1D array, cross-validated accuracy
    """
    scaler = StandardScaler()
    scaler.fit(feats.T)
    feats_stnd = scaler.transform(feats.T)
    feats_train, feats_test, labels_train, labels_test = train_test_split(feats_stnd, labels.T, test_size=hldout,
                                                                          random_state=23)
    logging.info(
        'Training SVM classifier on randomly partitioned {}% of training data...'.format((1 - hldout) * 100))
    classifier = svm.SVC(**svm_params)
    classifier.fit(feats_train, labels_train)
    logging.info('Done training SVM classifier mapping {} features to {} assignments.'.format(feats_train.shape,
                                                                                              labels_train.shape))
    logging.info('Predicting randomly sampled (non-overlapped) assignments '
                 'using the remaining {}%...'.format(HLDOUT * 100))
    scores = cross_val_score(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
    y_pred = cross_val_predict(classifier, feats_test, labels_test, cv=cv_it, n_jobs=-1)
    cm = confusion_matrix(labels_test, y_pred)
    logging.info('Done cross-validating the learned SVM classifier.'.format(feats_train.shape, labels_train.shape))
    return classifier, scores, cm


def main(train_folders):
    """
    Import training data, preprocess low likelihood
    """
    import bsoid_py.utils.likelihoodprocessing
    filenames, training_data, perc_rect = bsoid_py.utils.likelihoodprocessing.main(train_folders)

    """
    Extract features and train t-SNE, EM-GMM, and SVM on training set
    """
    f_10fps, trained_tsne = bsoid_tsne(training_data)
    gmm_assignments = bsoid_gmm(trained_tsne)
    classifier, scores, cm = bsoid_svm(f_10fps, gmm_assignments)

    """
    Plotting (True/False in LOCAL_CONFIG) some visuals
    and automatically saving .svg in the OUTPUT_PATH in GLOBAL_CONFIG
    """
    if PLOT_TRAINING:
        plot_classes(trained_tsne, gmm_assignments)
        plot_accuracy(scores)
        plot_cm(cm)
        plot_feats(f_10fps, gmm_assignments)
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores, cm
