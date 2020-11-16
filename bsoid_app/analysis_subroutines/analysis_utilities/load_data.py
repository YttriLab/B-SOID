import os

import joblib
import scipy.io


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
            data = joblib.load(fr)
        return [i for i in data]

    def load_feats(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_feats.sav'))), 'rb') as fr:
            data = joblib.load(fr)
        return [i for i in data]

    def load_embeddings(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_embeddings.sav'))), 'rb') as fr:
            data = joblib.load(fr)
        return [i for i in data]

    def load_clusters(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_clusters.sav'))), 'rb') as fr:
            data = joblib.load(fr)
        return [i for i in data]

    def load_classifier(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_randomforest.sav'))), 'rb') as fr:
            data = joblib.load(fr)
        return [i for i in data]

    def load_predictions(self):
        with open(os.path.join(self.path, str.join('', (self.name, '_predictions.sav'))), 'rb') as fr:
            data = joblib.load(fr)
        return [i for i in data]

