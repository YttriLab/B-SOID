import itertools
import math

from psutil import virtual_memory
import ffmpeg
import hdbscan
import joblib
import matplotlib as mpl
import networkx as nx
import streamlit as st
import umap
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing._data import StandardScaler

import bsoid_app
import bsoid_app.utils.statistics
from bsoid_app.classify import bsoid_extract, bsoid_predict
from bsoid_app.utils import likelihoodprocessing
from bsoid_app.utils.likelihoodprocessing import adp_filt, get_filenames, boxcar_center
from bsoid_app.utils.statistics import feat_dist
from bsoid_app.utils import statistics
from bsoid_app.utils.videoprocessing import *

# Intro
st.title('B-SOiD')
st.header('An open-source machine learning app for parsing (spatio-temporal) patterns.')
st.subheader(
    'Extract behavior from pose for any organism, any camera angle! Note that keeping the checkboxes unchecked when not needed speeds up the processing.')

demo_vids = {
    "Open-field, unrestrained, wild-type (Yttri lab @ CMU)": "./demo/ClusteredBehavior_aligned.mp4",
    "Open-field, tethered, OCD model (Ahmari lab @ UPitt)": "./demo/bsoid_grm_demo.mp4"
}
vid = st.selectbox("Notable examples, please contribute!", list(demo_vids.keys()), 0)
video_file = open(demo_vids[vid], 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

# Load previous run?
if st.sidebar.checkbox("Load previous run? This resumes training, or can load previously trained network for new analysis.", False):
    OUTPUT_PATH = st.sidebar.text_input('Enter the prior run output directory:')
    try:
        os.listdir(OUTPUT_PATH)
        st.markdown(
            'You have selected **{}** as your prior run root directory.'.format(OUTPUT_PATH))
    except FileNotFoundError:
        st.error('No such directory')
    MODEL_NAME = st.sidebar.text_input('Enter your prior run variable file prefix:')
    if MODEL_NAME:
        st.markdown('You have selected **{}_XXX.sav** as your prior variable files.'.format(str(MODEL_NAME)))
    else:
        st.error('Please enter a prefix name for prior run variable file.')
    last_run = True
else:
    last_run = False

if not last_run:
    # Setting things up
    # BASE_PATH, TRAIN_FOLDERS, FPS, OUTPUT_PATH and MODEL_NAME designations
    st.subheader('Find your data')
    st.write('The __BASE PATH__ contains multiple nested directories.')
    BASE_PATH = st.text_input('Enter a BASE PATH:')
    try:
        os.listdir(BASE_PATH)
        st.markdown(
            'You have selected **{}** as your root directory for training/testing sub-directories.'.format(BASE_PATH))
    except FileNotFoundError:
        st.error('No such directory')
    st.write('The __sub-directory(ies)__ each contain one or more .csv files. '
             'Currently supporting _2D_ and _single_ animal.')
    TRAIN_FOLDERS = []
    no_dir = int(st.number_input('How many BASE_PATH/SUB-DIRECTORIES for training?', value=3))
    st.markdown('Your will be training on **{}** csv containing sub-directories.'.format(no_dir))
    for i in range(no_dir):
        training_dir = st.text_input('Enter path to training directory NUMBER {} within base path:'.format(i + 1))
        try:
            os.listdir(str.join('', (BASE_PATH, training_dir)))
        except FileNotFoundError:
            st.error('No such directory')
        if not training_dir in TRAIN_FOLDERS:
            TRAIN_FOLDERS.append(training_dir)
    st.markdown('You have selected **sub-directory(ies)** *{}*.'.format(TRAIN_FOLDERS))
    st.write('Average __frame-rate__ for these processed .csv files. '
             'Your pose estimation will be integrated over 100ms. '
             'For most animal behaviors, static poses per 100ms appears to capture _sufficient information_ '
             'for behavioral clustering while maintaining _high temporal resolution._')
    FPS = int(st.number_input('What is your frame-rate?', value=60))
    st.markdown('Your framerate is **{}** frames per second.'.format(FPS))
    st.write('The __output directory__ will store B-SOID clustering _variable_ files and .csv _analyses_.')
    OUTPUT_PATH = st.text_input('Enter an output directory:')
    try:
        os.listdir(OUTPUT_PATH)
        st.markdown('You have selected **{}** to store results.'.format(str(OUTPUT_PATH)))
    except FileNotFoundError:
        st.error('No such directory, was there a typo or did you forget to create one?')
    st.write('For each run, computed variables are stored as __.sav files__. '
             'If you type in the same variable prefix as last run, your _workspace_ will be loaded.')
    MODEL_NAME = st.text_input('Enter a variable file name prefix:')
    if MODEL_NAME:
        st.markdown('You have named **{}_XXX.sav** as the variable files.'.format(str(MODEL_NAME)))
    else:
        st.error('Please enter a name for your variable file name prefix.')

    # Pre-processing
    st.subheader('__Pre-process__ the low-likelihood estimations as a representation of occlusion coordinates.')
    st.text_area('', '''
    Within each .csv file, the algorithm finds the best likelihood cutoff for each body part.
    ''')
    csv_rep = glob.glob(BASE_PATH + TRAIN_FOLDERS[0] + '/*.csv')
    curr_df = pd.read_csv(csv_rep[0], low_memory=False)
    currdf = np.array(curr_df)
    BP = st.multiselect('Body parts to include', [*currdf[0, 1:-1:3]], [*currdf[0, 1:-1:3]])
    BODYPARTS = []
    for b in BP:
        index = [i for i, s in enumerate(currdf[0, 1:]) if b in s]
        if not index in BODYPARTS:
            BODYPARTS += index
    BODYPARTS.sort()
    if st.button("Start pre-processing"):
        filenames = []
        rawdata_li = []
        data_li = []
        perc_rect_li = []
        for i, fd in enumerate(TRAIN_FOLDERS):  # Loop through folders
            f = get_filenames(BASE_PATH, fd)
            my_bar = st.progress(0)
            for j, filename in enumerate(f):
                curr_df = pd.read_csv(filename, low_memory=False)
                curr_df_filt, perc_rect = adp_filt(curr_df, BODYPARTS)
                rawdata_li.append(curr_df)
                perc_rect_li.append(perc_rect)
                data_li.append(curr_df_filt)
                my_bar.progress(round((j + 1) / len(f) * 100))
                filenames.append(filename)
        training_data = np.array(data_li)
        with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_data.sav'))), 'wb') as f:
            joblib.dump([BASE_PATH, FPS, BODYPARTS, filenames, rawdata_li, training_data, perc_rect_li], f)
        st.info('Processed a total of **{}** CSV files, and compiled into a **{}** data list.'.format(len(data_li),
                                                                                                      training_data.shape))
        st.balloons()
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_data.sav'))), 'rb') as fr:
        BASE_PATH, FPS, BODYPARTS, filenames, rawdata_li, training_data, perc_rect_li = joblib.load(fr)
    if st.checkbox('Show % body part processed per file?', False):
        st.write('This line chart shows __% body part below file-based threshold__')
        subllh_percent = pd.DataFrame(perc_rect_li)
        st.bar_chart(subllh_percent)
    # st.write('This allows you to scroll through and visualize raw vs processed data.')
    # if st.checkbox("Show raw & processed data?", False):
    #     try:
    #         ID = int(st.number_input('Enter csv/data-list index:', min_value=1, max_value=len(rawdata_li), value=1))
    #         st.markdown('This is file *{}*.'.format(filenames[ID - 1]))
    #         st.write(rawdata_li[ID - 1])
    #         st.write(training_data[ID - 1])
    #     except:
    #         pass

if last_run:
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_data.sav'))), 'rb') as fr:
        BASE_PATH, FPS, BODYPARTS, filenames, rawdata_li, training_data, perc_rect_li = joblib.load(fr)
    if st.checkbox('Show % body part processed per file?', False):
        st.write('This line chart shows __% body part below file-based threshold__')
        subllh_percent = pd.DataFrame(perc_rect_li)
        st.bar_chart(subllh_percent)
    st.markdown('**_CHECK POINT_**: Processed a total of **{}** CSV files, '
                'and compiled into a **{}** data list.'.format(len(rawdata_li), training_data.shape))
    st.write('This allows you to scroll through and visualize raw vs processed data.')
    if st.checkbox("Show raw & processed data?", False):
        try:
            ID = int(st.number_input('Enter csv/data-list index:', min_value=1, max_value=len(rawdata_li), value=1))
            st.write(rawdata_li[ID - 1])
            st.write(training_data[ID - 1])
        except:
            pass

# Feature extraction + UMAP
st.subheader('Perform __dimensionality reduction__ to improve clustering.')
st.text_area('', '''
For each body part, find the distance to all others, the angular change between these distances, and its displacement over time. 
That is A LOT of dimensions, so reducing it is necessary.
''')
if st.button("Start dimensionality reduction"):
    win_len = np.int(np.round(0.05 / (1 / FPS)) * 2 - 1)
    feats = []
    my_bar = st.progress(0)
    for m in range(len(training_data)):
        dataRange = len(training_data[m])
        dxy_r = []
        dis_r = []
        for r in range(dataRange):
            if r < dataRange - 1:
                dis = []
                for c in range(0, training_data[m].shape[1], 2):
                    dis.append(np.linalg.norm(training_data[m][r + 1, c:c + 2] - training_data[m][r, c:c + 2]))
                dis_r.append(dis)
            dxy = []
            for i, j in itertools.combinations(range(0, training_data[m].shape[1], 2), 2):
                dxy.append(training_data[m][r, i:i + 2] - training_data[m][r, j:j + 2])
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
        my_bar.progress(round((m + 1) / len(training_data) * 100))
    st.info('Done extracting features from a total of **{}** training CSV files.'.format(len(training_data)))
    for n in range(0, len(feats)):
        feats1 = np.zeros(len(training_data[n]))
        for k in range(round(FPS / 10), len(feats[n][0]), round(FPS / 10)):
            if k > round(FPS / 10):
                feats1 = np.concatenate((feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                         np.hstack((np.mean((feats[n][0:dxy_smth.shape[0],
                                                             range(k - round(FPS / 10), k)]), axis=1),
                                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                                            range(k - round(FPS / 10), k)]),
                                                           axis=1))).reshape(len(feats[0]), 1)), axis=1)
            else:
                feats1 = np.hstack((np.mean((feats[n][0:dxy_smth.shape[0], range(k - round(FPS / 10), k)]), axis=1),
                                    np.sum((feats[n][dxy_smth.shape[0]:feats[n].shape[0],
                                            range(k - round(FPS / 10), k)]), axis=1))).reshape(len(feats[0]), 1)
        if n > 0:
            f_10fps = np.concatenate((f_10fps, feats1), axis=1)
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = np.concatenate((f_10fps_sc, feats1_sc), axis=1)
        else:
            f_10fps = feats1
            scaler = StandardScaler()
            scaler.fit(feats1.T)
            feats1_sc = scaler.transform(feats1.T).T
            f_10fps_sc = feats1_sc  # scaling is important as I've seen wildly different stdev/feat between sessions
    feats_train = f_10fps_sc.T
    mem = virtual_memory()
    if mem.available > f_10fps_sc.shape[0] * f_10fps_sc.shape[1] * 32 * 100 + 256000000:
        trained_umap = umap.UMAP(n_neighbors=100,  # power law
                                 **UMAP_PARAMS).fit(feats_train)
    else:
        trained_umap = umap.UMAP(n_neighbors=100, low_memory=False,  # power law
                                 **UMAP_PARAMS).fit(feats_train)
    umap_embeddings = trained_umap.embedding_
    st.info(
        'Done non-linear transformation of **{}** instances from **{}** D into **{}** D.'.format(feats_train.shape[0],
                                                                                                 feats_train.shape[1],
                                                                                                 umap_embeddings.shape[
                                                                                                     1]))
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'wb') as f:
        joblib.dump([f_10fps, f_10fps_sc, umap_embeddings], f)
    st.balloons()

if last_run:
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    st.markdown('**_CHECK POINT_**: Done non-linear transformation of **{}** instances '
                'from **{}** D into **{}** D.'.format(f_10fps_sc.shape[1], f_10fps_sc.shape[0],
                                                      umap_embeddings.shape[1]))

# HDBSCAN
st.subheader('Perform density-based clustering.')
st.text_area('', '''
The following slider allows you to adjust cluster number.
The preset (0.5-1.5%) works for most large (> 25k instances) datasets. 
It is recommended to tweak this for cluster number > 40 or < 4.
''')
cluster_range = st.slider('Select range of minimum cluster size in %', 0.01, 5.0, (0.4, 1.2))
st.markdown('Your minimum cluster size ranges between **{}%** and **{}%**.'.format(cluster_range[0], cluster_range[1]))
if st.button("Start clustering"):
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    highest_numulab = -np.infty
    numulab = []
    min_cluster_range = np.linspace(cluster_range[0], cluster_range[1], 25)
    for min_c in min_cluster_range:
        trained_classifier = hdbscan.HDBSCAN(prediction_data=True,
                                             min_cluster_size=int(round(min_c * 0.01 * umap_embeddings.shape[0])),
                                             **HDBSCAN_PARAMS).fit(umap_embeddings)
        numulab.append(len(np.unique(trained_classifier.labels_)))
        if numulab[-1] > highest_numulab:
            st.info('Adjusting minimum cluster size to maximize cluster number...')
            highest_numulab = numulab[-1]
            best_clf = trained_classifier
    assignments = best_clf.labels_
    soft_clusters = hdbscan.all_points_membership_vectors(best_clf)
    soft_assignments = np.argmax(soft_clusters, axis=1)
    st.info('Done assigning labels for **{}** instances in **{}** D space'.format(*umap_embeddings.shape))
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'wb') as f:
        joblib.dump([assignments, soft_clusters, soft_assignments], f)
    st.balloons()

if last_run:
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)
    st.markdown('**_CHECK POINT_**: Done assigning labels for '
                '**{}** instances in **{}** D space'.format(*umap_embeddings.shape))

if st.checkbox("Show UMAP enhanced clustering plot?", True):
    st.write('Below are two cluster plots.')
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)
    fig1, plt1 = plot_classes(umap_embeddings[assignments >= 0], assignments[assignments >= 0])
    plt1.suptitle('HDBSCAN assignment')
    st.pyplot(fig1)
    st.write('The __soft__ assignment disregards noise and attempts to fit all data points to assignments '
             'based on highest probability.')
    fig2, plt2 = plot_classes(umap_embeddings[soft_assignments >= 0], soft_assignments[soft_assignments >= 0])
    plt2.suptitle('HDBSCAN soft assignment')
    st.pyplot(fig2)

st.subheader('Based on __soft__ assignment, train a neural network to _learn_ the rules.')
st.text_area('', '''
Neural network will be trained on recognizing distance, angles, and speed. 
This is for our vision in closed-loop experiments
             ''')
if st.button("Start training a behavioral neural network"):
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
        f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)
    feats_train, feats_test, labels_train, labels_test = train_test_split(f_10fps.T, soft_assignments.T,
                                                                          test_size=HLDOUT, random_state=23)
    st.info(
        'Training feedforward neural network on randomly partitioned {}% of training data...'.format(
            (1 - HLDOUT) * 100))
    classifier = MLPClassifier(**MLP_PARAMS)
    classifier.fit(feats_train, labels_train)
    clf = MLPClassifier(**MLP_PARAMS)
    clf.fit(f_10fps.T, soft_assignments.T)
    nn_assignments = clf.predict(f_10fps.T)
    st.info('Done training feedforward neural network '
            'mapping **{}** features to **{}** assignments.'.format(f_10fps.T.shape, soft_assignments.T.shape))
    scores = cross_val_score(classifier, feats_test, labels_test, cv=CV_IT, n_jobs=-1)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'wb') as f:
        joblib.dump([feats_test, labels_test, classifier, clf, scores, nn_assignments], f)
    st.balloons()

if last_run:
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as fr:
        feats_test, labels_test, classifier, clf, scores, nn_assignments = joblib.load(fr)
    st.markdown('**_CHECK POINT_**: Done training feedforward neural network '
                'mapping **{}** features to **{}** assignments.'.format(f_10fps.T.shape, soft_assignments.T.shape))

if st.checkbox("Show confusion matrix on {}% data?".format(HLDOUT * 100), False):
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as fr:
        feats_test, labels_test, classifier, clf, scores, nn_assignments = joblib.load(fr)
    np.set_printoptions(precision=2)
    titles_options = [("Non-normalized confusion matrix", None),
                      ("Normalized confusion matrix", 'true')]
    titlenames = [("counts"), ("norm")]
    j = 0
    st.write('Below are two confusion matrices - top: raw counts, bottom: probability. '
             'These matrices shows **true positives in diagonal**, false negatives in rows, and false positives in columns')
    for title, normalize in titles_options:
        cm = plot_confusion_matrix(classifier, feats_test, labels_test,
                                   cmap=plt.cm.Blues,
                                   normalize=normalize)
        cm.ax_.set_title(title)
        j += 1
        st.pyplot(cm.figure_)
    st.write(
        'If these are **NOT satisfactory**, either _increase_ the above minimum cluster size to remove noise subgroups, or include _more data_')
if st.checkbox("Show cross-validated accuracy on randomly selected {}% held-out test set?".format(HLDOUT * 100), False):
    st.write('For **overall** machine learning accuracy, a part of the error could be _cleaning up_ clustering noise.')
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as fr:
        feats_test, labels_test, classifier, clf, scores, nn_assignments = joblib.load(fr)
    fig, plt = plot_accuracy(scores)
    st.pyplot(fig)
    st.write(
        'If this is **NOT satisfactory**, either _increase_ the above minimum cluster size to remove noise subgroups, or include _more data_')

st.subheader('If reasonable/satisfied, you may export analyses results to {}'.format(OUTPUT_PATH))
txt5 = st.text_area('Result options descriptions:', '''
Input features: basic statistics of these extracted pairwise distance, angle, and speed features. 
Feature corresponding labels: these features time-locked to the labels. 
Soft assignment probabilities: if interested, the label probabilities of each time point.
''')
result1_options = st.multiselect('What type of results do you want to export',
                                 ['Input features', 'Feature corresponding labels', 'Soft assignment probabilities'],
                                 ['Feature corresponding labels'])
if st.button('Export'):
    if any('Input features' in o for o in result1_options):
        with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
            f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
        timestr = time.strftime("_%Y%m%d_%H%M")
        feat_range, feat_med, p_cts, edges = feat_dist(f_10fps)
        f_range_df = pd.DataFrame(feat_range, columns=['5%tile', '95%tile'])
        f_med_df = pd.DataFrame(feat_med, columns=['median'])
        f_pcts_df = pd.DataFrame(p_cts)
        f_pcts_df.columns = pd.MultiIndex.from_product([f_pcts_df.columns, ['prob']])
        f_edge_df = pd.DataFrame(edges)
        f_edge_df.columns = pd.MultiIndex.from_product([f_edge_df.columns, ['edge']])
        f_dist_data = pd.concat((f_range_df, f_med_df, f_pcts_df, f_edge_df), axis=1)
        f_dist_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('feature_distribution_10Hz', timestr, '.csv')))),
                           index=True, chunksize=10000, encoding='utf-8')
    if any('Feature corresponding labels' in o for o in result1_options):
        with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_feats.sav'))), 'rb') as fr:
            f_10fps, f_10fps_sc, umap_embeddings = joblib.load(fr)
        with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
            assignments, soft_clusters, soft_assignments = joblib.load(fr)
        with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as fr:
            feats_test, labels_test, classifier, clf, scores, nn_assignments = joblib.load(fr)
        timestr = time.strftime("_%Y%m%d_%H%M")
        length_nm = []
        angle_nm = []
        disp_nm = []
        for i, j in itertools.combinations(range(0, int(np.sqrt(f_10fps.shape[0]))), 2):
            length_nm.append(['distance between points:', i + 1, j + 1])
            angle_nm.append(['angular change for points:', i + 1, j + 1])
        for i in range(int(np.sqrt(f_10fps.shape[0]))):
            disp_nm.append(['displacement for point:', i + 1, i + 1])
        mcol = np.vstack((length_nm, angle_nm, disp_nm))
        feat_nm_df = pd.DataFrame(f_10fps.T, columns=mcol)
        umaphdb_data = np.concatenate([umap_embeddings, assignments.reshape(len(assignments), 1),
                                       soft_assignments.reshape(len(soft_assignments), 1),
                                       nn_assignments.reshape(len(nn_assignments), 1)], axis=1)
        micolumns = pd.MultiIndex.from_tuples([('UMAP embeddings', 'Dimension 1'), ('', 'Dimension 2'),
                                               ('', 'Dimension 3'), ('HDBSCAN', 'Assignment No.'),
                                               ('HDBSCAN*SOFT', 'Assignment No.'), ('Neural Net', 'Assignment No.')],
                                              names=['Type', 'Frame@10Hz'])
        umaphdb_df = pd.DataFrame(umaphdb_data, columns=micolumns)
        training_data = pd.concat((feat_nm_df, umaphdb_df), axis=1)
        soft_clust_prob = pd.DataFrame(soft_clusters)
        training_data.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('features_labels_10Hz', timestr, '.csv')))),
                             index=True, chunksize=10000, encoding='utf-8')
    if any('Soft assignment probabilities' in o for o in result1_options):
        with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
            assignments, soft_clusters, soft_assignments = joblib.load(fr)
        timestr = time.strftime("_%Y%m%d_%H%M")
        soft_clust_prob = pd.DataFrame(soft_clusters)
        soft_clust_prob.to_csv((os.path.join(OUTPUT_PATH, str.join('', ('soft_cluster_prob_10Hz', timestr, '.csv')))),
                               index=True, chunksize=10000, encoding='utf-8')
    st.balloons()

if st.sidebar.checkbox('Behavioral structure visual analysis?', False):
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_clusters.sav'))), 'rb') as fr:
        assignments, soft_clusters, soft_assignments = joblib.load(fr)
    with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_predictions.sav'))), 'rb') as fr:
        flders, flder, filenames, data_new, fs_labels = joblib.load(fr)
    selected_flder = st.sidebar.selectbox('select folder', [*flders])
    try:
        indices = [i for i, s in enumerate(flder) if str(selected_flder) in s]
        tm_c_all = []
        tm_p_all = []
        for idx in indices:
            runlen_df, dur_stats, B, df_tm, B_norm = statistics.main(fs_labels[idx], len(np.unique(soft_assignments)))
            tm_c_all.append(B)
            tm_p_all.append(B_norm)
        tm_c_ave = np.nanmean(tm_c_all, axis=0)
        tm_p_ave = np.nanmean(tm_p_all, axis=0)
        diag = [tm_c_ave[i][i] for i in range(len(tm_c_ave))]
        diag_p = np.array(diag) / np.array(diag).max()
        node_sizes = [50 * i for i in diag_p]
        A = np.matrix(tm_p_ave)
        np.fill_diagonal(A, 0)
        A_norm = A / A.sum(axis=1)
        where_are_NaNs = np.isnan(A_norm)
        A_norm[where_are_NaNs] = 0
        fig = plt.figure()
        G = nx.from_numpy_matrix(A_norm, create_using=nx.MultiDiGraph())
        pos = nx.layout.spring_layout(G)
        edge_colors = [G[u][v][0].get('weight') for u, v in G.edges()]
        nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue', with_label=True)
        edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                       arrowsize=8, edge_color=edge_colors,
                                       edge_cmap=plt.cm.Blues, width=1.5)
        lab_pos = [pos[i] + 0.005 for i in range(len(pos))]
        nx.draw_networkx_labels(G, lab_pos, font_size=10)
        pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
        pc.set_array(edge_colors)
        plt.colorbar(pc)
        ax = plt.gca()
        ax.set_axis_off()
        st.pyplot(fig)
    except:
        pass

else:
    st.subheader('Making sense of these behaviors and bulk process old/new data.')
    txt = st.text_area('Process flow options:', '''
    Generate predictions and corresponding videos: allows you to go video by video and analyze with visuals.
    Bulk process all csvs: once you have subjective definitions for labels, you can run predictions with high consistency. It will prompt for types of analysis to be exported.
    ''')

    pred_options = st.selectbox('Select an option:',
                                ('Generate predictions and corresponding videos', 'Bulk process all csvs'))
    if pred_options == 'Generate predictions and corresponding videos':
        csv_dir = st.text_input('Enter the testing data sub-directory within BASE PATH:')
        try:
            os.listdir(str.join('', (BASE_PATH, csv_dir)))
            st.markdown(
                'You have selected **{}** as your csv data sub-directory.'.format(csv_dir))
        except FileNotFoundError:
            st.error('No such directory')
        csv_file = st.selectbox('Select the csv file', sorted(os.listdir(str.join('', (BASE_PATH, csv_dir)))))
        vid_dir = st.text_input('Enter corresponding video directory (This can be outside of BASE PATH):')
        try:
            os.listdir(vid_dir)
            st.markdown(
                'You have selected **{}** as your video directory.'.format(vid_dir))
        except FileNotFoundError:
            st.error('No such directory')
        vid_file = st.selectbox('Select the video (.mp4 or .avi)', sorted(os.listdir(vid_dir)))
        st.markdown('You have selected **{}** as your video matching **{}**.'.format(vid_file, csv_file))
        csvname = os.path.basename(csv_file).rpartition('.')[0]
        try:
            os.mkdir(str.join('', (BASE_PATH, csv_dir, '/pngs')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (BASE_PATH, csv_dir, '/pngs', '/', csvname)))
        except FileExistsError:
            pass
        frame_dir = str.join('', (BASE_PATH, csv_dir, '/pngs', '/', csvname))
        st.markdown('You have created **{}** as your PNG directory for video {}.'.format(frame_dir, vid_file))
        probe = ffmpeg.probe(os.path.join(vid_dir, vid_file))
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        num_frames = int(video_info['nb_frames'])
        bit_rate = int(video_info['bit_rate'])
        avg_frame_rate = round(int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
        if st.button('Start frame extraction for {} frames at {} frames per second'.format(num_frames, avg_frame_rate)):
            try:
                (ffmpeg.input(os.path.join(vid_dir, vid_file))
                 .filter('fps', fps=avg_frame_rate)
                 .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                         s=str.join('', (str(int(width * 0.5)), 'x', str(int(height * 0.5)))), sws_flags='bilinear',
                         start_number=0)
                 .run(capture_stdout=True, capture_stderr=True))
                st.info('Done extracting **{}** frames from video **{}**.'.format(num_frames, vid_file))
            except ffmpeg.Error as e:
                print('stdout:', e.stdout.decode('utf8'))
                print('stderr:', e.stderr.decode('utf8'))
        try:
            os.mkdir(str.join('', (BASE_PATH, csv_dir, '/mp4s')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (BASE_PATH, csv_dir, '/mp4s', '/', csvname)))
        except FileExistsError:
            pass
        shortvid_dir = str.join('', (BASE_PATH, csv_dir, '/mp4s', '/', csvname))
        st.markdown('You have created **{}** as your .mp4 directory '
                    'for group examples from video {}.'.format(shortvid_dir, vid_file))
        min_time = st.number_input('Enter minimum time for bout in ms:', value=100)
        min_frames = round(float(min_time) * 0.001 * float(FPS))
        st.markdown('You have entered **{} ms** as your minimum duration per bout, '
                    'which is equivalent to **{} frames**.'
                    '(drop this down for more group representations)'.format(min_time, min_frames))
        number_examples = st.slider('Select number of non-repeated examples', 1, 10, 3)
        st.markdown('Your will obtain a maximum of **{}** non-repeated output examples per group.'.format(number_examples))
        out_fps = int(st.number_input('Enter output frame-rate:', value=30))
        playback_speed = float(out_fps) / float(FPS)
        st.markdown('Your have selected to view these examples at **{} FPS**, '
                    'which is equivalent to **{}X speed**.'.format(out_fps, playback_speed))
        if st.button("Predict labels and create example videos"):
            with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as fr:
                feats_test, labels_test, classifier, clf, scores, nn_assignments = joblib.load(fr)
            curr_df = pd.read_csv(os.path.join(str.join('', (BASE_PATH, csv_dir, '/', csv_file))), low_memory=False)
            curr_df_filt, perc_rect = adp_filt(curr_df, BODYPARTS)
            test_data = [curr_df_filt]
            labels_fs = []
            labels_fs2 = []
            fs_labels = []
            for i in range(0, len(test_data)):
                feats_new = bsoid_extract(test_data, FPS)
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
                for l in range(math.floor(FPS / 10)):
                    labels_fs2.append(labels_fs[k][l])
                fs_labels.append(np.array(labels_fs2).flatten('F'))
            st.info('Done frameshift-predicting **{}**.'.format(csv_file))
            create_labeled_vid(fs_labels[0], int(min_frames), int(number_examples), int(out_fps),
                               frame_dir, shortvid_dir)
            st.balloons()
        if st.checkbox("Show example videos? (loading it up from {})".format(shortvid_dir), False):
            example_vid = st.selectbox('Select the video (.mp4 or .avi)', sorted(os.listdir(shortvid_dir)))
            example_vid_file = open(os.path.join(str.join('', (shortvid_dir, '/', example_vid))), 'rb')
            st.markdown('You have selected **{}** as your video from {}.'.format(example_vid, shortvid_dir))
            video_bytes = example_vid_file.read()
            st.video(video_bytes)

    if pred_options == 'Bulk process all csvs':
        st.write('Bulk processing will take some time for large datasets.'
                 'This includes a lot of files, long videos, and/or high frame-rates.')
        TEST_FOLDERS = []
        no_dir = int(st.number_input('How many sub-directories for bulk predictions?', value=3))
        st.markdown('Your will be predicting on **{}** csv containing sub-directories.'.format(no_dir))
        for i in range(no_dir):
            test_dir = st.text_input('Enter path to test directory number {} within base path:'.format(i + 1))
            try:
                os.listdir(str.join('', (BASE_PATH, test_dir)))
            except FileNotFoundError:
                st.error('No such directory')
            if not test_dir in TEST_FOLDERS:
                TEST_FOLDERS.append(test_dir)
        st.markdown('You have selected sub-directory(ies) **{}**.'.format(TEST_FOLDERS))
        FPS = int(st.number_input('What is your framerate for these csvs?', value=60))
        st.markdown('Your framerate is **{}** frames per second for these csvs.'.format(FPS))
        st.text_area('Select the analysis of interest to you. If in doubt, select all.', '''
        Predicted labels with original pose: labels written into original .csv files (time-locked).
        Behavioral bout lengths in chronological order: the behaviors and its bouts over time. 
        Behavioral bout statistics: basic statistics for these behavioral durations. 
        Transition matrix: behavioral transitions based on Markov Decision Process.
        ''')
        result2_options = st.multiselect('What type of results do you want to export?',
                                         ['Predicted labels with original pose',
                                          'Behavioral bout lengths in chronological order',
                                          'Behavioral bout statistics', 'Transition matrix'],
                                         ['Predicted labels with original pose', 'Behavioral bout statistics'])
        if st.button("Begin bulk csv processing, potentially a long computation"):
            st.write('These B-SOiD csv files will be saved in the original pose estimation csv containing folders, under sub-directory BSOID.')
            with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_neuralnet.sav'))), 'rb') as fr:
                feats_test, labels_test, classifier, clf, scores, nn_assignments = joblib.load(fr)
            flders, filenames, data_new, perc_rect = likelihoodprocessing.main(BASE_PATH, TEST_FOLDERS, BODYPARTS)
            labels_fs = []
            labels_fs2 = []
            fs_labels = []
            bar = st.progress(0)
            for i in range(0, len(data_new)):
                feats_new = bsoid_extract([data_new[i]], FPS)
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
                bar.progress(round((i + 1) / len(data_new) * 100))
            for k in range(0, len(labels_fs)):
                labels_fs2 = []
                for l in range(math.floor(FPS / 10)):
                    labels_fs2.append(labels_fs[k][l])
                fs_labels.append(np.array(labels_fs2).flatten('F'))
            st.info('Done frameshift-predicting a total of **{}** files.'.format(len(data_new)))
            filenames = []
            all_df = []
            flder = []
            for i, fd in enumerate(TEST_FOLDERS):  # Loop through folders
                f = get_filenames(BASE_PATH, fd)
                for j, filename in enumerate(f):
                    curr_df = pd.read_csv(filename, low_memory=False)
                    filenames.append(filename)
                    flder.append(fd)
                    all_df.append(curr_df)
            for i in range(0, len(fs_labels)):
                timestr = time.strftime("_%Y%m%d_%H%M_")
                csvname = os.path.basename(filenames[i]).rpartition('.')[0]
                fs_labels_pad = np.pad(fs_labels[i], (0, len(all_df[i]) - 2 - len(fs_labels[i])), 'edge')
                df2 = pd.DataFrame(fs_labels_pad, columns={'B-SOiD labels'})
                df2.loc[len(df2)] = ''
                df2.loc[len(df2)] = ''
                df2 = df2.shift()
                df2.loc[0] = ''
                df2 = df2.shift()
                df2.loc[0] = ''
                frames = [df2, all_df[0]]
                xyfs_df = pd.concat(frames, axis=1)
                runlen_df, dur_stats, B, df_tm, B_norm = statistics.main(fs_labels[i], len(np.unique(nn_assignments)))
                try:
                    os.mkdir(str.join('', (BASE_PATH, flder[i], '/BSOID')))
                except FileExistsError:
                    pass
                if any('Predicted labels with original pose' in o for o in result2_options):
                    xyfs_df.to_csv(os.path.join(
                        str.join('', (BASE_PATH, flder[i], '/BSOID')),
                        str.join('', ('labels_pose_', str(FPS), 'Hz', timestr, csvname, '.csv'))),
                        index=True, chunksize=10000, encoding='utf-8')
                if any('Behavioral bout lengths in chronological order' in o for o in result2_options):
                    runlen_df.to_csv(os.path.join(
                        str.join('', (BASE_PATH, flder[i], '/BSOID')),
                        str.join('', ('bout_lengths_', str(FPS), 'Hz', timestr, csvname, '.csv'))),
                        index=True, chunksize=10000, encoding='utf-8')
                if any('Behavioral bout statistics' in o for o in result2_options):
                    dur_stats.to_csv(os.path.join(
                        str.join('', (BASE_PATH, flder[i], '/BSOID')),
                        str.join('', ('bout_stats_', str(FPS), 'Hz', timestr, csvname, '.csv'))),
                    index=True, chunksize=10000, encoding='utf-8')
                if any('Transition matrix' in o for o in result2_options):
                    df_tm.to_csv(os.path.join(
                        str.join('', (BASE_PATH, flder[i], '/BSOID')),
                        str.join('', ('transitions_mat_', str(FPS), 'Hz', timestr, csvname,'.csv'))),
                    index=True, chunksize=10000, encoding='utf-8')
            with open(os.path.join(OUTPUT_PATH, str.join('', (MODEL_NAME, '_predictions.sav'))), 'wb') as f:
                joblib.dump([flders, flder, filenames, data_new, fs_labels], f)
            st.balloons()
