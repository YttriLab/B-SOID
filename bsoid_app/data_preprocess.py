import os
from datetime import date

import h5py
import joblib
import randfacts
import streamlit as st

from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *


class preprocess:

    def __init__(self):
        st.subheader('LOAD DATA and PREPROCESS')
        self.pose_chosen = []
        self.input_filenames = []
        self.raw_input_data = []
        self.processed_input_data = []
        self.sub_threshold = []
        self.software = st.selectbox('What type of __pose-estimation software__?',
                                     ('DeepLabCut', 'SLEAP', 'OpenPose'))
        if self.software == 'DeepLabCut':
            self.ftype = st.selectbox('What type of __output file__?',
                                      ('csv', 'h5'))
        if self.software == 'SLEAP':
            self.ftype = 'h5'
            st.write('Currently only supporting {} type files'.format(self.ftype))
        if self.software == 'OpenPose':
            self.ftype = 'json'
            st.write('Currently only supporting {} type files'.format(self.ftype))
        st.write('Select the pose estimate root directory containing '
                 '1 or more xxx.{} containing sub-directories.'.format(self.ftype))
        self.root_path = st.text_input('Enter a __root directory__, e.g. __/Users/projectX__', os.getcwd())
        try:
            os.listdir(self.root_path)
            st.markdown(
                'You have selected **{}** as your _root directory_'.format(self.root_path))
        except FileNotFoundError:
            st.error('No such directory')
        st.write(
            'Select the pose estimate containing sub-directories, e.g. /control, under root directory. '
            'Currently supporting _2D_ and _single_ animal.')
        self.data_directories = []
        no_dir = int(st.number_input('How many __data containing directories__ '
                                     'under {} for training?'.format(self.root_path), value=3))
        st.markdown('Your will be training on *{}* data file containing sub-directories.'.format(no_dir))
        for i in range(no_dir):
            d = str.join('', ('/', st.selectbox('Enter # {} __data file containing directory__ under {}, '
                                                'e.g. __/control__ for /Users/projectX/control/xxx.{}'
                                                ''.format(i + 1, self.root_path, self.ftype),
                                                (os.listdir(self.root_path)), index=0)))
            try:
                os.listdir(str.join('', (self.root_path, d)))
            except FileNotFoundError:
                st.error('No such directory')
            if not d in self.data_directories:
                self.data_directories.append(d)
        st.markdown('You have selected **{}** as your _sub-directory(ies)_.'.format(self.data_directories))
        st.write('Average video frame-rate for xxx.{} pose estimate files.'.format(self.ftype))
        self.framerate = int(st.number_input('What is your frame-rate?', value=60))
        st.markdown('You have selected **{} frames per second**.'.format(self.framerate))
        st.write('Select a working directory for B-SOiD')
        self.working_dir = st.text_input('Enter a __working directory__, e.g. __/Users/projectX/output__',
                                         str.join('', (self.root_path, '/output')))
        try:
            os.listdir(self.working_dir)
            st.markdown('You have selected **{}** for B-SOiD working directory.'.format(self.working_dir))
        except FileNotFoundError:
            st.error('Cannot access working directory, was there a typo or did you forget to create one?')
        st.write('Input a prefix name for B-SOiD variables.')
        st.text('')
        st.write('*CAUTION*: It will OVERWRITE same prefix.')
        today = date.today()
        d4 = today.strftime("%b-%d-%Y")
        self.prefix = st.text_input('Enter a __variable filename__ prefix, e.g. __control_sessions_2020__', d4)
        if self.prefix:
            st.markdown('You have decided on **{}** as the prefix.'.format(self.prefix))
        else:
            st.error('Please enter a prefix.')

    def compile_data(self):
        st.write('Identify pose to include in clustering.')
        if self.software == 'DeepLabCut' and self.ftype == 'csv':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.csv')
            file0_df = pd.read_csv(data_files[0], low_memory=False)
            file0_array = np.array(file0_df)
            p = st.multiselect('Identified __pose__ to include:', [*file0_array[0, 1:-1:3]], [*file0_array[0, 1:-1:3]])
            for a in p:
                index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if st.button("__Preprocess__"):
                funfacts = randfacts.getFact()
                st.info(str.join('', ('Preprocessing... Here is a random fact: ', funfacts)))
                for i, fd in enumerate(self.data_directories):  # Loop through folders
                    f = get_filenames(self.root_path, fd)
                    my_bar = st.progress(0)
                    for j, filename in enumerate(f):
                        file_j_df = pd.read_csv(filename, low_memory=False)
                        file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                        self.raw_input_data.append(file_j_df)
                        self.sub_threshold.append(p_sub_threshold)
                        self.processed_input_data.append(file_j_processed)
                        self.input_filenames.append(filename)
                        my_bar.progress(round((j + 1) / len(f) * 100))
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                    )
                st.info('Processed a total of **{}** .{} files, and compiled into a '
                        '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                   np.array(self.processed_input_data).shape))
                st.balloons()
        elif self.software == 'DeepLabCut' and self.ftype == 'h5':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.h5')
            file0_df = pd.read_hdf(data_files[0], low_memory=False)
            p = st.multiselect('Identified __pose__ to include:',
                               [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])],
                               [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])])
            for a in p:
                index = [i for i, s in enumerate(np.array(file0_df.columns.get_level_values(1))) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if st.button("__Preprocess__"):
                funfacts = randfacts.getFact()
                st.info(str.join('', ('Preprocessing... Here is a random fact: ', funfacts)))
                for i, fd in enumerate(self.data_directories):
                    f = get_filenamesh5(self.root_path, fd)
                    my_bar = st.progress(0)
                    for j, filename in enumerate(f):
                        file_j_df = pd.read_hdf(filename, low_memory=False)
                        file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen)
                        self.raw_input_data.append(file_j_df)
                        self.sub_threshold.append(p_sub_threshold)
                        self.processed_input_data.append(file_j_processed)
                        self.input_filenames.append(filename)
                        my_bar.progress(round((j + 1) / len(f) * 100))
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                    )
                st.info('Processed a total of **{}** .{} files, and compiled into a '
                        '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                   np.array(self.processed_input_data).shape))
                st.balloons()
        elif self.software == 'SLEAP' and self.ftype == 'h5':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.h5')
            file0_df = h5py.File(data_files[0], 'r')
            p = st.multiselect('Identified __pose__ to include:',
                               [*np.array(file0_df['node_names'][:])],
                               [*np.array(file0_df['node_names'][:])])
            for a in p:
                index = [i for i, s in enumerate(np.array(file0_df['node_names'][:])) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if st.button("__Preprocess__"):
                funfacts = randfacts.getFact()
                st.info(str.join('', ('Preprocessing... Here is a random fact: ', funfacts)))
                for i, fd in enumerate(self.data_directories):
                    f = get_filenamesh5(self.root_path, fd)
                    my_bar = st.progress(0)
                    for j, filename in enumerate(f):
                        file_j_df = h5py.File(filename, 'r')
                        file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
                        self.raw_input_data.append(file_j_df['tracks'][:][0])
                        self.sub_threshold.append(p_sub_threshold)
                        self.processed_input_data.append(file_j_processed)
                        self.input_filenames.append(filename)
                        my_bar.progress(round((j + 1) / len(f) * 100))
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                    )
                st.info('Processed a total of **{}** .{} files, and compiled into a '
                        '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                   np.array(self.processed_input_data).shape))
                st.balloons()
        elif self.software == 'OpenPose' and self.ftype == 'json':
            data_files = glob.glob(self.root_path + self.data_directories[0] + '/*.json')
            file0_df = read_json_single(data_files[0])
            file0_array = np.array(file0_df)
            p = st.multiselect('Identified __pose__ to include:', [*file0_array[0, 1:-1:3]], [*file0_array[0, 1:-1:3]])
            for a in p:
                index = [i for i, s in enumerate(file0_array[0, 1:]) if a in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
            if st.button("__Preprocess__"):
                funfacts = randfacts.getFact()
                st.info(str.join('', ('Preprocessing... Here is a random fact: ', funfacts)))
                for i, fd in enumerate(self.data_directories):
                    f = get_filenamesjson(self.root_path, fd)
                    json2csv_multi(f)
                    filename = f[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
                    file_j_df = pd.read_csv(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')),
                                            low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                    self.raw_input_data.append(file_j_df)
                    self.sub_threshold.append(p_sub_threshold)
                    self.processed_input_data.append(file_j_processed)
                    self.input_filenames.append(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')))
                with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_data.sav'))), 'wb') as f:
                    joblib.dump(
                        [self.root_path, self.data_directories, self.framerate, self.pose_chosen, self.input_filenames,
                         self.raw_input_data, np.array(self.processed_input_data), self.sub_threshold], f
                    )
                st.info('Processed a total of **{}** .{} files, and compiled into a '
                        '**{}** data list.'.format(len(self.processed_input_data), self.ftype,
                                                   np.array(self.processed_input_data).shape))
                st.balloons()

    def show_bar(self):
        visuals.plot_bar(self.sub_threshold)

    def show_data_table(self):
        visuals.show_data_table(self.raw_input_data, self.processed_input_data)
