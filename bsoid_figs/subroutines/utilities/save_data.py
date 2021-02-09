import scipy.io
import os
import joblib


class results:

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def save_sav(self, datalist, fname):
        with open(os.path.join(self.path, str.join('', (self.name, '_', fname, '.sav'))), 'wb') as f:
            joblib.dump(datalist, f)
