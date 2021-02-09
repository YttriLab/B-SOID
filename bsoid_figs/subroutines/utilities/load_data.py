import scipy.io
import os
import joblib


def load_mat(file):
    return scipy.io.loadmat(file)


def load_sav(path, name, fname):
    with open(os.path.join(path, str.join('', (name, '_', fname, '.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


class appdata:

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def load_data(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_data.sav'))), 'rb') as fr:
            BASE_PATH, TRAIN_FOLDERS, FPS, BODYPARTS, filenames, \
            rawdata_li, training_data, perc_rect_li = joblib.load(fr)
        return BASE_PATH, TRAIN_FOLDERS, FPS, BODYPARTS, filenames, rawdata_li, training_data, perc_rect_li

    def load_feats(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_feats.sav'))), 'rb') as fr:
            f_10fps, f_10fps_sc = joblib.load(fr)
        return f_10fps, f_10fps_sc

    def load_embeddings(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_embeddings.sav'))), 'rb') as fr:
            f_10fps_sub, train_embeddings = joblib.load(fr)
        return f_10fps_sub, train_embeddings

    def load_clusters(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_clusters.sav'))), 'rb') as fr:
            min_cluster_range, assignments, soft_clusters, soft_assignments = joblib.load(fr)
        return min_cluster_range, assignments, soft_clusters, soft_assignments

    def load_classifier(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_randomforest.sav'))), 'rb') as fr:
            feats_test, labels_test, classifier, clf, scores, nn_assignments = joblib.load(fr)
        return feats_test, labels_test, classifier, clf, scores, nn_assignments

    def load_predictions(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_predictions.sav'))), 'rb') as fr:
            flders, flder, filenames, data_new, fs_labels = joblib.load(fr)
        return flders, flder, filenames, data_new, fs_labels

