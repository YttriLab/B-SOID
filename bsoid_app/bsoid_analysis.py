import streamlit as st

from analysis_subroutines import video_analysis, machine_performance, trajectory_analysis, \
    kinematics_analysis, directed_graph_analysis
from analysis_subroutines.analysis_utilities.cache_workspace import load_data
from analysis_subroutines.analysis_utilities.visuals import *
from bsoid_utilities.load_css import local_css


def streamlit_run(pyfile):
    os.system("streamlit run {}.py".format(pyfile))


st.set_page_config(page_title='B-SOiD anaylsis', page_icon="📊",
                   layout='wide', initial_sidebar_state='auto')
local_css("./bsoid_app/bsoid_utilities/style.css")
title = "<div> <span class='bold'><span class='h1'>B-SOID</span></span> " \
        "   <span class='h2'>anaylsis 📊</span></span> </div>"
st.markdown(title, unsafe_allow_html=True)
st.markdown('Step 1: Pick the directory and workspace to analyze.')
st.markdown('Step 2: Once input, select the type of results to analyze using the sidebar modules.')
st.text('')
working_dir = st.text_input('Enter B-SOiD __output directory__ from using the B-SOiD --version 2.0 App')
try:
    os.listdir(working_dir)
    st.markdown(
        'You have selected **{}** as your B-SOiD App results run root directory.'.format(working_dir))
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

[framerate, features, sampled_features, sampled_embeddings, assignments, soft_assignments,
 folders, folder, filenames, new_data, new_predictions] = load_data(working_dir, prefix)
if st.sidebar.checkbox('Synchronized B-SOiD video (paper Supp. Video 1)', False, key='v'):
    video_generator = video_analysis.bsoid_video(working_dir, prefix, features, sampled_features,
                                                 sampled_embeddings, soft_assignments, framerate,
                                                 filenames, new_data)
    video_generator.main()
if st.sidebar.checkbox('K-fold accuracy boxplot (paper fig2c)', False, key='a'):
    performance_eval = machine_performance.performance(working_dir, prefix, soft_assignments)
    performance_eval.main()
if st.sidebar.checkbox('Limb trajectories (paper fig2d/g)', False, key='t'):
    st.write(filenames[0].partition('.')[-1])
    trajectory_mapper = trajectory_analysis.trajectory(working_dir, prefix, soft_assignments, framerate,
                                                       filenames, new_data, new_predictions)
    trajectory_mapper.main()
if st.sidebar.checkbox('(beta) Kinematics (paper fig6b/d)', False, key='k'):
    kinematics_analyzer = kinematics_analysis.kinematics(working_dir, prefix, framerate, soft_assignments, filenames)
    kinematics_analyzer.main()
if st.sidebar.checkbox('(alpha) Behavioral directed graph', False, key='d'):
    network = directed_graph_analysis.directed_graph(working_dir, prefix, soft_assignments,
                                                     folders, folder, new_predictions)
    network.main()
if st.sidebar.checkbox('Return computation to main app (please close current browser when new browser pops up)', False):
    streamlit_run('./bsoid_app')

