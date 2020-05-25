"""
Classify behaviors based on (x,y) using trained B-SOiD behavioral model.
B-SOiD behavioral model has been developed using bsoid_app.main.build()
"""

import math

import itertools
import numpy as np

from bsoid_app.utils import videoprocessing
from bsoid_app.utils.likelihoodprocessing import boxcar_center
from bsoid_app.utils.visuals import *


def bsoid_extract(data, fps):
    """
    Extracts features based on (x,y) positions
    :param data: list, csv data
    :param fps: scalar, input for camera frame-rate
    :return f_10fps: 2D array, extracted features
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
    f_10fps = []
    for n in range(0, len(feats)):
        feats1 = np.zeros(len(data[n]))
        for s in range(math.floor(fps / 10)):
            for k in range(round(fps / 10) + s, len(feats[n][0]), round(fps / 10)):
                    if k > round(fps / 10) + s:
                        feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                                 np.hstack((np.mean((feats[n][0:dxy_smth.shape[0],
                                                                     range(k - round(fps / 10), k)]), axis=1),
                                                            np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                                    range(k - round(fps / 10), k)]),
                                                                   axis=1))).reshape(len(feats[0]), 1)), axis=1)
                    else:
                        feats1 = np.hstack((np.mean((feats[n][0:dxy_smth.shape[0], range(k - round(fps / 10), k)]),
                                                    axis=1),
                                            np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                    range(k - round(fps / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
            logging.info('Done integrating features into 100ms bins from CSV file {}.'.format(n + 1))
            f_10fps.append(feats1)
    return f_10fps


def bsoid_predict(feats, clf):
    """
    :param feats: list, multiple feats (original feature space)
    :param clf: Obj, MLP classifier
    :return nonfs_labels: list, label/100ms
    """
    labels_fslow = []
    for i in range(0, len(feats)):
        labels = clf.predict(feats[i].T)
        logging.info('Done predicting file {} with {} instances in {} D space.'.format(i + 1, feats[i].shape[1],
                                                                                       feats[i].shape[0]))
        labels_fslow.append(labels)
    logging.info('Done predicting a total of {} files.'.format(len(feats)))
    return labels_fslow


def bsoid_frameshift(data_new, fps, clf):
    """
    Frame-shift paradigm to output behavior/frame
    :param data_new: list, new data from predict_folders
    :param fps: scalar, argument specifying camera frame-rate in LOCAL_CONFIG
    :param clf: Obj, MLP classifier
    :return fs_labels, 1D array, label/frame
    """
    labels_fs = []
    labels_fs2 = []
    labels_fshigh = []
    for i in range(0, len(data_new)):
        data_offset = []
        for j in range(math.floor(fps / 10)):
            data_offset.append(data_new[i][j:, :])
        feats_new = bsoid_extract(data_offset)
        labels = bsoid_predict(feats_new, clf)
        for m in range(0, len(labels)):
            labels[m] = labels[m][::-1]
        labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
        for n, l in enumerate(labels):
            labels_pad[n][0:len(l)] = l
            labels_pad[n] = labels_pad[n][::-1]
            if n > 0:
                labels_pad[n][0:n] = labels_pad[n - 1][0:n]
        labels_fs.append(labels_pad.astype(int))
    for k in range(0, len(labels_fs)):
        labels_fs2 = []
        for l in range(math.floor(fps / 10)):
            labels_fs2.append(labels_fs[k][l])
        labels_fshigh.append(np.array(labels_fs2).flatten('F'))
    logging.info('Done frameshift-predicting a total of {} files.'.format(len(data_new)))
    return labels_fshigh

