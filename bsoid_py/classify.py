"""
Classify behaviors based on (x,y) using trained B-SOiD behavioral model.
B-SOiD behavioral model has been developed using bsoid_py.main.build()
"""

import math

import numpy as np

from bsoid_py.utils import videoprocessing
from bsoid_py.utils.likelihoodprocessing import boxcar_center
from bsoid_py.utils.visuals import *


def bsoid_extract(data, bodyparts=BODYPARTS, fps=FPS):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param bodyparts: dict, body parts with their orders
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
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
            sn_disp[k] = np.linalg.norm(data[m][k + 1, 2 * bodyparts.get('Snout/Head'):2 * bodyparts.get('Snout/Head') + 1] -
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
    f_10fps = []
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
        f_10fps.append(feats1)
    return f_10fps


def bsoid_predict(feats, model):
    """
    :param feats: list, multiple feats (original feature space)
    :param model: Obj, MLP classifier
    :return labels_fslow: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        logging.info('Predicting file {} with {} instances '
                     'using learned classifier: {}{}...'.format(i+1, feats[i].shape[1], 'bsoid_', MODEL_NAME))
        labels = model.predict(feats[i].T)
        logging.info('Done predicting file {} with {} instances in {} D space.'.format(i+1, feats[i].shape[1],
                                                                                       feats[i].shape[0]))
        labels_fslow.append(labels)
    logging.info('Done predicting a total of {} files.'.format(len(feats)))
    return labels_fslow


def bsoid_frameshift(data_new, fps, model):
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param model: Obj, MLP classifier
    :return labels_fshigh, 1D array, label/frame
    """
    labels_fs = []
    labels_fs2 = []
    labels_fshigh = []
    for i in range(0, len(data_new)):
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract(data_offset)
        labels = bsoid_predict(feats_new, model)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key = lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n-1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(0, len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(fps / 10)):
            labels_fs2.append(labels_fs[k][l])
        labels_fshigh.append(np.array(labels_fs2).flatten('F'))
    logging.info('Done frameshift-predicting a total of {} files.'.format(len(data_new)))
    return labels_fshigh


def main(predict_folders, fps, behv_model):
    """
    :param predict_folders: list, data folders
    :param fps: scalar, camera frame-rate
    :behv_model: object, MLP classifier
    :return data_new: list, csv data
    :return feats_new: 2D array, extracted features
    :return labels_fslow, 1D array, label/100ms
    :return labels_fshigh, 1D array, label/frame
    """
    import bsoid_py.utils.likelihoodprocessing
    filenames, data_new, perc_rect = bsoid_py.utils.likelihoodprocessing.main(predict_folders)
    feats_new = bsoid_extract(data_new)
    labels_fslow = bsoid_predict(feats_new, behv_model)
    labels_fshigh = bsoid_frameshift(data_new, fps, behv_model)
    if PLOT_TRAINING:
        plot_feats(feats_new, labels_fslow)
    if GEN_VIDEOS:
        videoprocessing.main(VID_NAME, labels_fslow[ID], FPS, FRAME_DIR)
    return data_new, feats_new, labels_fslow, labels_fshigh

