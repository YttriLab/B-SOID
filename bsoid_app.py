from streamlit import caching

from bsoid_app import data_preprocess, extract_features, clustering, machine_learner, \
    export_training, video_creator, predict
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_css import local_css
from bsoid_app.bsoid_utilities.load_workspace import *


def streamlit_run(pyfile):
    os.system("streamlit run {}.py".format(pyfile))


st.set_page_config(page_title='B-SOiD v2.0', page_icon="🐁",
                   layout='wide', initial_sidebar_state='auto')
local_css("bsoid_app/bsoid_utilities/style.css")
title = "<div> <span class='bold'><span class='h1'>B-SOID</span></span> " \
        "   <span class='h2'>--version 2.0 🐁</span> </div>"
st.markdown(title, unsafe_allow_html=True)
st.markdown('Step 1: Select Load Data and Preprocess. Complete the step.')
st.markdown('Step 2: Deselect Load Data and Preprocess, and select Load Previous Iteration. Fill in prompts.')
st.markdown('Step 3: Starting with Extract Features and Identify Clusters, select single procedure and progress.')
st.text('')
if st.sidebar.checkbox('Load previous iteration', False, key='l'):
    working_dir, prefix = query_workspace()
if st.sidebar.checkbox('Load data and preprocess', False, key='d'):
    try:
        [_, _, _, _, _, raw_input_data, processed_input_data, sub_threshold] = load_data(working_dir, prefix)
        st.markdown('**_CHECK POINT_**: Processed a total of **{}** data files, '
                    'and compiled into a **{}** data list. Move on to '
                    '__Extract and embed features__.'.format(len(raw_input_data), processed_input_data.shape))
        if st.checkbox('Redo?', False):
            caching.clear_cache()
            processor = data_preprocess.preprocess()
            processor.compile_data()
        if st.checkbox('Show % time (possibly) occluded?', True):
            visuals.plot_bar(sub_threshold)
        if st.checkbox("Show raw vs processed data?", False):
            visuals.show_data_table(raw_input_data, processed_input_data)
    except NameError:
        processor = data_preprocess.preprocess()
        processor.compile_data()
if st.sidebar.checkbox('Extract and embed features', False, key='f'):
    [_, _, framerate, _, _, _, processed_input_data, _] = load_data(working_dir, prefix)
    extractor = extract_features.extract(working_dir, prefix, processed_input_data, framerate)
    extractor.main()
if st.sidebar.checkbox('Identify and tweak number of clusters', False, key='c'):
    [_, sampled_embeddings] = load_embeddings(working_dir, prefix)
    clusterer = clustering.cluster(working_dir, prefix, sampled_embeddings)
    clusterer.main()
if st.sidebar.checkbox('(Optional) What did B-SOiD learn?', False, key='e'):
    [sampled_features, _] = load_embeddings(working_dir, prefix)
    [_, assignments, assign_prob, soft_assignments] = load_clusters(working_dir, prefix)
    exporter = export_training.export(working_dir, prefix, sampled_features,
                                      assignments, assign_prob, soft_assignments)
    exporter.save_csv()
if st.sidebar.checkbox('Create a model', False, key='t'):
    [features, _] = load_feats(working_dir, prefix)
    [sampled_features, _] = load_embeddings(working_dir, prefix)
    [_, assignments, _, _] = load_clusters(working_dir, prefix)
    learning_protocol = machine_learner.protocol(working_dir, prefix, features, sampled_features, assignments)
    learning_protocol.main()
if st.sidebar.checkbox('Generate video snippets for interpretation', False, key='g'):
    [root_path, data_directories, framerate, pose_chosen, input_filenames, _, processed_input_data, _] \
        = load_data(working_dir, prefix)
    [_, _, _, clf, _, _] = load_classifier(working_dir, prefix)
    creator = video_creator.creator(root_path, data_directories, processed_input_data, pose_chosen,
                                    working_dir, prefix, framerate, clf, input_filenames)
    creator.main()
if st.sidebar.checkbox('Predict old/new files using a model', False, key='p'):
    [root_path, data_directories, framerate, pose_chosen, input_filenames, _, processed_input_data, _] \
        = load_data(working_dir, prefix)
    [_, _, _, clf, _, predictions] = load_classifier(working_dir, prefix)
    predictor = predict.prediction(root_path, data_directories, input_filenames, processed_input_data, working_dir,
                                   prefix, framerate, pose_chosen, predictions, clf)
    predictor.main()
if st.sidebar.checkbox('Load up analysis app (please close current browser when new browser pops up)', False):
    streamlit_run('./bsoid_app/bsoid_analysis')
