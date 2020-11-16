import streamlit as st
import os
import joblib


@st.cache
def load_data(path, name):
    with open(os.path.join(path, str.join('', (name, '_data.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


def query_workspace():
    working_dir = st.sidebar.text_input('Enter the prior B-SOiD working directory:')
    try:
        os.listdir(working_dir)
        st.markdown(
            'You have selected **{}** for prior working directory.'.format(working_dir))
    except FileNotFoundError:
        st.error('No such directory')
    files = [i for i in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, i)) and \
             '_data.sav' in i and not '_accuracy' in i and not '_coherence' in i]
    bsoid_variables = [files[i].partition('_data.sav')[0] for i in range(len(files))]
    bsoid_prefix = []
    for var in bsoid_variables:
        if var not in bsoid_prefix:
            bsoid_prefix.append(var)
    prefix = st.selectbox('Select prior B-SOiD prefix', bsoid_prefix)
    try:
        st.markdown('You have selected **{}_XXX.sav** for prior prefix.'.format(prefix))
    except TypeError:
        st.error('Please input a prior prefix to load workspace.')
    return working_dir, prefix


@st.cache
def load_feats(path, name):
    with open(os.path.join(path, str.join('', (name, '_feats.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


@st.cache
def load_embeddings(path, name):
    with open(os.path.join(path, str.join('', (name, '_embeddings.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


@st.cache
def load_clusters(path, name):
    with open(os.path.join(path, str.join('', (name, '_clusters.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


@st.cache(allow_output_mutation=True)
def load_classifier(path, name):
    with open(os.path.join(path, str.join('', (name, '_randomforest.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


@st.cache
def load_predictions(path, name):
    with open(os.path.join(path, str.join('', (name, '_predictions.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]


def load_new_feats(path, name):
    with open(os.path.join(path, str.join('', (name, '_new_feats.sav'))), 'rb') as fr:
        data = joblib.load(fr)
    return [i for i in data]




