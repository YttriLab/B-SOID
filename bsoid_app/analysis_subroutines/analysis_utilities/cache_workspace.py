import os

import joblib
import streamlit as st


@st.cache
def load_data(path, name):
    with open(os.path.join(path, str.join('', (name, '_data.sav'))), 'rb') as fr:
        _, _, framerate, _, _, _, _, _ = joblib.load(fr)
    with open(os.path.join(path, str.join('', (name, '_feats.sav'))), 'rb') as fr:
        features, _ = joblib.load(fr)
    with open(os.path.join(path, str.join('', (name, '_embeddings.sav'))), 'rb') as fr:
        sampled_features, sampled_embeddings = joblib.load(fr)
    with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
        _, assignments, _, soft_assignments = joblib.load(fr)
    with open(os.path.join(path, str.join('', (name, '_predictions.sav'))), 'rb') as fr:
        folders, folder, filenames, new_data, new_predictions = joblib.load(fr)
    return framerate, features, sampled_features, sampled_embeddings, assignments, soft_assignments, \
           folders, folder, filenames, new_data, new_predictions
