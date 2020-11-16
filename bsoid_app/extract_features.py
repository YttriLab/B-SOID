import itertools
import math
import os

import joblib
import numpy as np
import randfacts
import streamlit as st
import umap
from psutil import virtual_memory
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit import caching

from bsoid_app.bsoid_utilities.likelihoodprocessing import boxcar_center
from bsoid_app.bsoid_utilities.load_workspace import load_feats, load_embeddings
from bsoid_app.config import *


class extract:

    def __init__(self, working_dir, prefix, processed_input_data, framerate):
        st.subheader('EXTRACT AND EMBED FEATURES')
        self.working_dir = working_dir
        self.prefix = prefix
        self.processed_input_data = processed_input_data
        self.framerate = framerate
        self.train_size = []
        self.features = []
        self.scaled_features = []
        self.sampled_features = []
        self.sampled_embeddings = []

    def subsample(self):
        data_size = 0
        for n in range(len(self.processed_input_data)):
            data_size += len(range(round(self.framerate / 10), self.processed_input_data[n].shape[0],
                                   round(self.framerate / 10)))
        fraction = st.number_input('Enter training input __fraction__ (do not change this value if you wish '
                                   'to generate the side-by-side video seen on our GitHub page):',
                                   min_value=0.1, max_value=1.0, value=1.0)
        if fraction == 1.0:
            self.train_size = data_size
        else:
            self.train_size = int(data_size * fraction)
        st.markdown('You have opted to train on a cumulative of **{} minutes** total. '
                    'If this does not sound right, the framerate might be wrong.'.format(self.train_size / 600))

    def compute(self):
        if st.button("__Extract Features__"):
            funfacts = randfacts.getFact()
            st.info(str.join('', ('Extracting... Here is a random fact: ', funfacts)))
            try:
                [self.features, self.scaled_features] = load_feats(self.working_dir, self.prefix)
            except:
                window = np.int(np.round(0.05 / (1 / self.framerate)) * 2 - 1)
                f = []
                my_bar = st.progress(0)
                for n in range(len(self.processed_input_data)):
                    data_n_len = len(self.processed_input_data[n])
                    dxy_list = []
                    disp_list = []
                    for r in range(data_n_len):
                        if r < data_n_len - 1:
                            disp = []
                            for c in range(0, self.processed_input_data[n].shape[1], 2):
                                disp.append(
                                    np.linalg.norm(self.processed_input_data[n][r + 1, c:c + 2] -
                                                   self.processed_input_data[n][r, c:c + 2]))
                            disp_list.append(disp)
                        dxy = []
                        for i, j in itertools.combinations(range(0, self.processed_input_data[n].shape[1], 2), 2):
                            dxy.append(self.processed_input_data[n][r, i:i + 2] -
                                       self.processed_input_data[n][r, j:j + 2])
                        dxy_list.append(dxy)
                    disp_r = np.array(disp_list)
                    dxy_r = np.array(dxy_list)
                    disp_boxcar = []
                    dxy_eu = np.zeros([data_n_len, dxy_r.shape[1]])
                    ang = np.zeros([data_n_len - 1, dxy_r.shape[1]])
                    dxy_boxcar = []
                    ang_boxcar = []
                    for l in range(disp_r.shape[1]):
                        disp_boxcar.append(boxcar_center(disp_r[:, l], window))
                    for k in range(dxy_r.shape[1]):
                        for kk in range(data_n_len):
                            dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                            if kk < data_n_len - 1:
                                b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                                a_3d = np.hstack([dxy_r[kk, k, :], 0])
                                c = np.cross(b_3d, a_3d)
                                ang[kk, k] = np.dot(np.dot(np.sign(c[2]), 180) / np.pi,
                                                    math.atan2(np.linalg.norm(c),
                                                               np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :])))
                        dxy_boxcar.append(boxcar_center(dxy_eu[:, k], window))
                        ang_boxcar.append(boxcar_center(ang[:, k], window))
                    disp_feat = np.array(disp_boxcar)
                    dxy_feat = np.array(dxy_boxcar)
                    ang_feat = np.array(ang_boxcar)
                    f.append(np.vstack((dxy_feat[:, 1:], ang_feat, disp_feat)))
                    my_bar.progress(round((n + 1) / len(self.processed_input_data) * 100))
                for m in range(0, len(f)):
                    f_integrated = np.zeros(len(self.processed_input_data[m]))
                    for k in range(round(self.framerate / 10), len(f[m][0]), round(self.framerate / 10)):
                        if k > round(self.framerate / 10):
                            f_integrated = np.concatenate(
                                (f_integrated.reshape(f_integrated.shape[0], f_integrated.shape[1]),
                                 np.hstack((np.mean((f[m][0:dxy_feat.shape[0],
                                                     range(k - round(self.framerate / 10), k)]), axis=1),
                                            np.sum((f[m][dxy_feat.shape[0]:f[m].shape[0],
                                                    range(k - round(self.framerate / 10), k)]), axis=1)
                                            )).reshape(len(f[0]), 1)), axis=1
                            )
                        else:
                            f_integrated = np.hstack(
                                (np.mean((f[m][0:dxy_feat.shape[0], range(k - round(self.framerate / 10), k)]), axis=1),
                                 np.sum((f[m][dxy_feat.shape[0]:f[m].shape[0],
                                         range(k - round(self.framerate / 10), k)]), axis=1))).reshape(len(f[0]), 1)
                    if m > 0:
                        self.features = np.concatenate((self.features, f_integrated), axis=1)
                        scaler = StandardScaler()
                        scaler.fit(f_integrated.T)
                        scaled_f_integrated = scaler.transform(f_integrated.T).T
                        self.scaled_features = np.concatenate((self.scaled_features, scaled_f_integrated), axis=1)
                    else:
                        self.features = f_integrated
                        scaler = StandardScaler()
                        scaler.fit(f_integrated.T)
                        scaled_f_integrated = scaler.transform(f_integrated.T).T
                        self.scaled_features = scaled_f_integrated
                self.features = np.array(self.features)
                self.scaled_features = np.array(self.scaled_features)
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_feats.sav'))), 'wb') as f:
                    joblib.dump([self.features, self.scaled_features], f)
            st.info('Done extracting features from a total of **{}** training data files. '
                    'Now reducing dimensions...'.format(len(self.processed_input_data)))
            self.learn_embeddings()

    def learn_embeddings(self):
        input_feats = self.scaled_features.T
        pca = PCA()
        pca.fit(self.scaled_features.T)
        num_dimensions = np.argwhere(np.cumsum(pca.explained_variance_ratio_) >= 0.7)[0][0] + 1
        if self.train_size > input_feats.shape[0]:
            self.train_size = input_feats.shape[0]
        np.random.seed(0)
        sampled_input_feats = input_feats[np.random.choice(input_feats.shape[0], self.train_size, replace=False)]
        features_transposed = self.features.T
        np.random.seed(0)
        self.sampled_features = features_transposed[np.random.choice(features_transposed.shape[0],
                                                                     self.train_size, replace=False)]
        st.info('Randomly sampled **{} minutes**... '.format(self.train_size / 600))
        mem = virtual_memory()
        available_mb = mem.available >> 20
        st.write('You have {} MB RAM 🐏 available'.format(available_mb))
        if available_mb > (sampled_input_feats.shape[0] * sampled_input_feats.shape[1] * 32 * 60) / 1024 ** 2 + 64:
            st.write('RAM 🐏 available is sufficient')
            try:
                learned_embeddings = umap.UMAP(n_neighbors=60, n_components=num_dimensions,
                                               **UMAP_PARAMS).fit(sampled_input_feats)
            except:
                st.error('Failed on feature embedding. Try again by unchecking sidebar and rerunning extract features.')
        else:
            st.info(
                'Detecting that you are running low on available memory for this computation, '
                'setting low_memory so will take longer.')
            try:
                learned_embeddings = umap.UMAP(n_neighbors=60, n_components=num_dimensions, low_memory=True,
                                               **UMAP_PARAMS).fit(sampled_input_feats)
            except:
                st.error('Failed on feature embedding. Try again by unchecking sidebar and rerunning extract features.')
        self.sampled_embeddings = learned_embeddings.embedding_
        st.info(
            'Done non-linear embedding of {} instances from **{}** D into **{}** D.'.format(
                *self.sampled_features.shape, self.sampled_embeddings.shape[1]))
        with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_embeddings.sav'))), 'wb') as f:
            joblib.dump([self.sampled_features, self.sampled_embeddings], f)
        st.balloons()

    def main(self):
        try:
            [self.sampled_features, self.sampled_embeddings] = load_embeddings(self.working_dir, self.prefix)
            st.markdown('**_CHECK POINT_**: Done non-linear transformation of **{}** instances '
                        'from **{}** D into **{}** D. Move on to __Identify and '
                        'tweak number of clusters__'.format(*self.sampled_features.shape, self.sampled_embeddings.shape[1]))
            if st.checkbox('Redo?', False, key='er'):
                caching.clear_cache()
                self.subsample()
                self.compute()
        except FileNotFoundError:
            self.subsample()
            self.compute()
