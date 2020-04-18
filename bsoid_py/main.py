"""
A master that runs BOTH
1. Training a unsupervised machine learning model based on patterns in spatio-temporal (x,y) changes.
2. Predicting new behaviors using (x,y) based on learned classifier.
"""

from bsoid_py.config import *
import joblib
import os
import pandas as pd
import time
import glob


def build(train_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
    Automatically saves single CSV file containing training outputs (in 10Hz, 100ms per row):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. embedded features (number of training data points by 3 dimensions, columns 8-10)
    3. em-gmm assignments (number of training data points by 1, columns 11)
    Automatically saves classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    """
    import bsoid_py.train
    f_10fps, trained_tsne, gmm_assignments, classifier, scores, cm = bsoid_py.train.main(train_folders)
    alldata = np.concatenate([f_10fps.T, trained_tsne, gmm_assignments.reshape(len(gmm_assignments), 1)], axis=1)
    micolumns = pd.MultiIndex.from_tuples([('Features', 'Relative snout to forepaws placement'),
                                           ('', 'Relative snout to hind paws placement'),
                                           ('', 'Inter-forepaw distance'),
                                           ('', 'Body length'), ('', 'Body angle'), ('', 'Snout displacement'),
                                           ('', 'Tail-base displacement'),
                                           ('Embedded t-SNE', 'Dimension 1'), ('', 'Dimension 2'),
                                           ('', 'Dimension 3'), ('EM-GMM', 'Assignment No.')],
                                          names=['Type', 'Frame@10Hz'])
    training_data = pd.DataFrame(alldata, columns=micolumns)
    timestr = time.strftime("_%Y%m%d_%H%M")
    training_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_train10Hz', timestr, '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
    with open(os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', MODEL_NAME, timestr, '.sav'))), 'wb') as f:
        joblib.dump(classifier, f)
    logging.info('Saved.')
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores, cm


def run(predict_folders):
    """
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns labels_fslow, labels_fshigh: see bsoid_py.classify
    Automatically loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing new outputs (1 in 10Hz, 1 in FPS, both with same format):
    1. original features (number of training data points by 7 dimensions, columns 1-7)
    2. SVM predicted labels (number of training data points by 1, columns 8)
    """
    import bsoid_py.classify
    from bsoid_py.utils.likelihoodprocessing import get_filenames
    import bsoid_py.utils.statistics
    with open(os.path.join(OUTPUT_PATH, FINALMODEL_NAME), 'rb') as fr:
        behv_model = joblib.load(fr)
    data_new, feats_new, labels_fslow, labels_fshigh = bsoid_py.classify.main(predict_folders, FPS, behv_model)
    filenames = []
    allDf = []
    for i, fd in enumerate(predict_folders):  # Loop through folders
        f = get_filenames(fd)
        for j, filename in enumerate(f):
            logging.info('Importing CSV file {} from folder {}'.format(j + 1, i + 1))
            currDf = pd.read_csv(filename, low_memory=False)
            filenames.append(filename)
            allDf.append(currDf)
    for i in range(0, len(feats_new)):
        alldata = np.concatenate([feats_new[i].T, labels_fslow[i].reshape(len(labels_fslow[i]), 1)], axis=1)
        micolumns = pd.MultiIndex.from_tuples([('Features', 'Relative snout to forepaws placement'),
                                               ('', 'Relative snout to hind paws placement'),
                                               ('', 'Inter-forepaw distance'),
                                               ('', 'Body length'), ('', 'Body angle'), ('', 'Snout displacement'),
                                               ('', 'Tail-base displacement'), ('Multi-class SVM', 'B-SOiD labels')],
                                              names=['Type', 'Frame@10Hz'])
        predictions = pd.DataFrame(alldata, columns=micolumns)
        timestr = time.strftime("_%Y%m%d_%H%M")
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        predictions.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_10Hz_', csvname, timestr, '.csv')))),
                           index=True, chunksize=10000, encoding='utf-8')
        dur_stats1, df_tm1 = bsoid_py.utils.statistics.main(labels_fslow[i], OUTPUT_PATH)
        dur_stats1.to_csv((os.path.join(output_path, str.join('', ('bsoid_10Hz_stats_', csvname, timestr, '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        df_tm1.to_csv((os.path.join(output_path, str.join('', ('bsoid_10Hz_transitions_', csvname, timestr, '.csv')))),
                     index=True, chunksize=10000, encoding='utf-8')
        labels_fshigh_pad = np.pad(labels_fshigh[i], (6, 0), 'edge')
        df2 = pd.DataFrame(labels_fshigh_pad, columns={'B-SOiD labels'})
        df2.loc[len(df2)] = ''
        df2.loc[len(df2)] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        df2 = df2.shift()
        df2.loc[0] = ''
        frames = [allDf[0], df2]
        xyfs_df = pd.concat(frames, axis=1)
        csvname = os.path.basename(filenames[i]).rpartition('.')[0]
        xyfs_df.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('bsoid_', str(FPS), 'Hz_', csvname, timestr, '.csv')))),
                       index=True, chunksize=10000, encoding='utf-8')
        dur_stats2, df_tm2 = bsoid_py.utils.statistics.main(labels_fshigh[i], OUTPUT_PATH)
        dur_stats2.to_csv((os.path.join(output_path, str.join('', ('bsoid_stats_', str(FPS), 'Hz_', csvname, timestr,
                                                                  '.csv')))),
                         index=True, chunksize=10000, encoding='utf-8')
        df_tm2.to_csv((os.path.join(output_path, str.join('', ('bsoid_transitions_', str(FPS), 'Hz_', csvname, timestr,
                                                              '.csv')))),
                     index=True, chunksize=10000, encoding='utf-8')
    logging.info('All saved.')
    return data_new, feats_new, labels_fslow, labels_fshigh


def main(train_folders, predict_folders):
    """
    :param train_folders: list, folders to build behavioral model on
    :param predict_folders: list, folders to run prediction using behavioral model
    :returns f_10fps, trained_tsne, gmm_assignments, classifier, scores: see bsoid_py.train
    :returns feats_new, labels_fslow, labels_fshigh: see bsoid_py.classify
    Automatically saves and loads classifier in OUTPUTPATH with MODELNAME in LOCAL_CONFIG
    Automatically saves CSV files containing training and new outputs
    """
    f_10fps, trained_tsne, gmm_assignments, classifier, scores, cm = build(train_folders)
    data_new, feats_new, labels_fslow, labels_fshigh = run(predict_folders)
    return f_10fps, trained_tsne, gmm_assignments, classifier, scores, cm, feats_new, labels_fslow, labels_fshigh


if __name__ == "__main__":
    f_10fps, trained_tsne, gmm_assignments, classifier, scores, cm, feats_new, labels_fslow, labels_fshigh \
        = main(TRAIN_FOLDERS, PREDICT_FOLDERS)
