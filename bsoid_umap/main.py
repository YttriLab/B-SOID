"""
A master that runs BOTH
1. Training a unsupervised + supervised machine learning model based on patterns in spatio-temporal (x,y) changes.
2. Predicting new behaviors using (x,y) based on learned classifier.
"""

import os
import time

import itertools
import joblib
import numpy as np
import pandas as pd

from bsoid_umap.config import *


def build(train_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments: see bsoid_umap.train
    Automatically saves single CSV file containing training outputs.
    Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    """
    import bsoid_umap.train
    from bsoid_umap.utils.statistics import feat_dist
    f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, \
    nn_assignments = bsoid_umap.train.main(train_folders)
    timestr = time.strftime("_%Y%m%d_%H%M")
    feat_range, feat_med, p_cts, edges = feat_dist(f_10fps)
    f_range_df = pd.DataFrame(feat_range, columns=['5%tile', '95%tile'])
    f_med_df = pd.DataFrame(feat_med, columns=['median'])
    f_pcts_df = pd.DataFrame(p_cts)
    f_pcts_df.columns = pd.MultiIndex.from_product([f_pcts_df.columns, ['prob']])
    f_edge_df = pd.DataFrame(edges)
    f_edge_df.columns = pd.MultiIndex.from_product([f_edge_df.columns, ['edge']])
    f_dist_data = pd.concat((f_range_df, f_med_df, f_pcts_df, f_edge_df), axis=1)
    f_dist_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_featdist_10Hz', timestr, '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
    length_nm = []
    angle_nm = []
    disp_nm = []
    for i, j in itertools.combinations(range(0, int(np.sqrt(f_10fps.shape[0]))), 2):
        length_nm.append(['distance between points:', i+1, j+1])
        angle_nm.append(['angular change for points:', i+1, j+1])
    for i in range(int(np.sqrt(f_10fps.shape[0]))):
        disp_nm.append(['displacement for point:', i+1, i+1])
    mcol = np.vstack((length_nm, angle_nm, disp_nm))
    feat_nm_df = pd.DataFrame(f_10fps.T, columns=mcol)
    umaphdb_data = np.concatenate([umap_embeddings, hdb_assignments.reshape(len(hdb_assignments), 1),
                              soft_assignments.reshape(len(soft_assignments), 1),
                              nn_assignments.reshape(len(nn_assignments), 1)], axis=1)
    micolumns = pd.MultiIndex.from_tuples([('UMAP embeddings', 'Dimension 1'), ('', 'Dimension 2'),
                                           ('', 'Dimension 3'), ('HDBSCAN', 'Assignment No.'),
                                           ('HDBSCAN*SOFT', 'Assignment No.'), ('Neural Net', 'Assignment No.')],
                                          names=['Type', 'Frame@10Hz'])
    umaphdb_df = pd.DataFrame(umaphdb_data, columns=micolumns)
    training_data = pd.concat((feat_nm_df, umaphdb_df), axis=1)
    soft_clust_prob = pd.DataFrame(soft_clusters)
    training_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_trainlabels_10Hz', timestr, '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
    soft_clust_prob.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labelprob_10Hz', timestr, '.csv')))),
                           index=True, chunksize=10000, encoding='utf-8')
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', MODEL_NAME, '.sav'))), 'wb') as f:
        joblib.dump([f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters,
                     nn_classifier, scores, nn_assignments], f)
    logging.info('Saved.')
    return f_10fps, f_10fps_sc, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, \
           scores, nn_assignments


# def retrain(train_folders):
#     """
#     :param train_folders: list, folders to build behavioral model on
#     :returns f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments: see bsoid_umap.train
#     Automatically saves single CSV file containing training outputs.
#     Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
#     """
#     with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', MODEL_NAME, '.sav'))), 'rb') as fr:
#         f_10fps, umap_embeddings, hdb_assignments, soft_assignments, soft_clusters, nn_classifier, scores, \
#         nn_assignments = joblib.load(fr)
#     from bsoid_umap.utils.videoprocessing import vid2frame
#     vid2frame(VID_NAME, f_10fps[ID], FPS, FRAME_DIR)
#     labels_df = pd.read_csv('/Users/ahsu/Sign2Speech/Notebook/labels.csv', low_memory=False)
#
#     import bsoid_umap.retrain
#     f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments = bsoid_umap.train.main(train_folders)
#     alldata = np.concatenate([umap_embeddings, nn_assignments.reshape(len(nn_assignments), 1)], axis=1)
#     micolumns = pd.MultiIndex.from_tuples([('UMAP embeddings', 'Dimension 1'), ('', 'Dimension 2'),
#                                            ('', 'Dimension 3'), ('Neural Net', 'Assignment No.')],
#                                           names=['Type', 'Frame@10Hz'])
#     training_data = pd.DataFrame(alldata, columns=micolumns)
#     timestr = time.strftime("_%Y%m%d_%H%M")
#     training_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_trainlabels_10Hz', timestr, '.csv')))),
#                          index=True, chunksize=10000, encoding='utf-8')
#     with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', MODEL_NAME, '.sav'))), 'wb') as f:
#         joblib.dump([f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments], f)
#     logging.info('Saved.')
#     return f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments


def run(predict_folders):
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns data_new, fs_labels: see bsoid_umap.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs.
    """
    import bsoid_umap.classify
    from bsoid_umap.utils.likelihoodprocessing import get_filenames
    import bsoid_umap.utils.statistics
    from bsoid_umap.utils.visuals import plot_tmat
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', MODEL_NAME, '.sav'))), 'rb') as fr:
        f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments = joblib.load(fr)
    data_new, fs_labels = bsoid_umap.classify.main(predict_folders, FPS, nn_classifier)
    filenames = []
    all_df = []
    for i, fd in enumerate(predict_folders):  # Loop through folders
        f = get_filenames(fd)
        for j, filename in enumerate(f):
            logging.info('Importing CSV file {} from folder {}'.format(j + 1, i + 1))
            curr_df = pd.read_csv(filename, low_memory=False)
            filenames.append(filename)
            all_df.append(curr_df)
    for i in range(0, len(fs_labels)):
        timestr = time.strftime("_%Y%m%d_%H%M")
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        fs_labels_pad = np.pad(fs_labels[i], (6, 0), 'edge')
        df2 = pd.DataFrame(fs_labels_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [df2, all_df[0]]
        xyfs_df = pd.concat(frames, axis=1)
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_labels_', str(FPS), 'Hz', timestr, csvname,
                                                                '.csv')))),
                       index=True, chunksize=10000, encoding='utf-8')
        runlen_df, dur_stats, df_tm = bsoid_umap.utils.statistics.main(fs_labels[i])
        runlen_df.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_runlen_', str(FPS), 'Hz', timestr, csvname,
                                                                  '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        dur_stats.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_stats_', str(FPS), 'Hz', timestr, csvname,
                                                                  '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        df_tm.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_transitions_', str(FPS), 'Hz', timestr, csvname,
                                                              '.csv')))),
                     index=True, chunksize=10000, encoding='utf-8')
        if PLOT:
            fig = plot_tmat(df_tm)
            my_file = 'transition_matrix'
            fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, str(FPS), 'Hz', timestr, csvname, '.svg'))))
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_predictions', timestr, '.sav'))), 'wb') as f:
        joblib.dump([data_new, fs_labels], f)
    logging.info('All saved.')
    return data_new, fs_labels


def main(train_folders, predict_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments: see bsoid_umap.train
    :returns feats_new, fs_labels: see bsoid_umap.classify
    Automatically saves and loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing training and new outputs
    """
    f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments = build(train_folders)
    data_new, fs_labels = run(predict_folders)
    return f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments, data_new, fs_labels


if __name__ == "__main__":
    f_10fps, umap_embeddings, nn_classifier, scores, nn_assignments, \
    data_new, fs_labels = main(TRAIN_FOLDERS, PREDICT_FOLDERS)
