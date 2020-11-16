import os
from datetime import date

import h5py
import joblib
import streamlit as st

from bsoid_app.bsoid_utilities import statistics
from bsoid_app.bsoid_utilities.bsoid_classification import *
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *


class prediction:

    def __init__(self, root_path, data_directories, input_filenames, processed_input_data, working_dir, prefix,
                 framerate, pose_chosen, predictions, clf):
        st.subheader('PREDICT OLD/NEW FILES USING A MODEL')
        st.markdown('This could take some time for large datasets.')
        self.options = st.multiselect('What csv files to export?',
                                      ['Labels tagged onto pose files', 'Group durations (in frames)',
                                       'Transition matrix'],
                                      ['Labels tagged onto pose files', 'Transition matrix'])
        self.root_path = root_path
        self.data_directories = data_directories
        self.input_filenames = input_filenames
        self.processed_input_data = processed_input_data
        self.working_dir = working_dir
        self.prefix = prefix
        self.framerate = framerate
        self.pose_chosen = pose_chosen
        self.predictions = predictions
        self.clf = clf
        self.use_train = []
        self.new_root_path = []
        self.new_directories = []
        self.filetype = []
        self.new_prefix = []
        self.new_framerate = []
        self.new_data = []
        self.new_features = []
        self.nonfs_predictions = []
        self.folders = []
        self.folder = []
        self.filenames = []
        self.new_predictions = []
        self.all_df = []

    def setup(self):
        if st.checkbox('All {} training folders containing '
                       'a total of {} files?'.format(len(self.data_directories),
                                                     self.processed_input_data.shape[0]), False, key='pt'):
            self.new_root_path = self.root_path
            self.filetype = [s for i, s in enumerate(['csv', 'h5', 'json'])
                             if s in self.input_filenames[0].partition('.')[-1]][0]
            self.new_directories = self.data_directories
            self.new_framerate = self.framerate
            self.new_prefix = self.prefix
        else:
            st.write(
                'Select the pose estimate containing sub-directories, e.g. /control, under root directory. '
                'Currently supporting _2D_ and _single_ animal.')
            if st.checkbox('New root directory (not {}) for these new files?'.format(self.root_path), False, key='pn'):
                self.new_root_path = st.text_input('Enter a __root directory__, e.g. __/Users/projectX__', os.getcwd())
            else:
                self.new_root_path = self.root_path
            num_dir = int(st.number_input('How many __data containing directories__ '
                                          'under {} for B-SOiD predictions?'.format(self.new_root_path), value=3))
            st.markdown('Your will predict on data files in *{}* directories.'.format(num_dir))
            self.filetype = st.selectbox('What type of file are these new data?', ('csv', 'h5', 'json'),
                                         index=int([i for i, s in enumerate(['csv', 'h5', 'json'])
                                                    if s in self.input_filenames[0].partition('.')[-1]][0]))
            for i in range(num_dir):
                new_directory = st.text_input('Enter # {} __data file containing directory__ under {}, '
                                              'e.g. __/control__ for /Users/projectX/control/xxx.{}'
                                              ''.format(i + 1, self.new_root_path, self.filetype))
                try:
                    os.listdir(str.join('', (self.new_root_path, new_directory)))
                except FileNotFoundError:
                    st.error('No such directory')
                if not new_directory in self.new_directories:
                    self.new_directories.append(new_directory)
            st.markdown('You have selected **{}** as your _sub-directory(ies)_.'.format(self.new_directories))
            st.write('Average video frame-rate for xxx.{} pose estimate files.'.format(self.filetype))
            self.new_framerate = int(st.number_input('What is your frame-rate?', value=self.framerate))
            st.markdown('You have selected **{} frames per second**.'.format(self.new_framerate))
        if st.checkbox('For every new dataset, you would want to change the prefix so it does not overwrite previous '
                       'predictions. Would you like to change the prefix? Currently it is set to save'
                       ' as **{}/{}_predictions.sav**.'.format(self.working_dir, self.prefix), False, key='pp'):
            today = date.today()
            d4 = today.strftime("%b-%d-%Y")
            self.new_prefix = st.text_input('Enter new prediction variable prefix:', d4)
            if self.new_prefix:
                st.markdown('You have chosen **{}_predictions.sav** for new predictions.'.format(self.new_prefix))
            else:
                st.error('Please enter a name for your new prediction variable prefix.')
        else:
            self.new_prefix = self.prefix

    def predict(self):
        if st.button("Predict labels"):
            st.markdown('These files will be saved in {}/_your_data_folder_x_/BSOID'.format(self.new_root_path))
            if self.filetype == 'csv':
                for i, fd in enumerate(self.new_directories):
                    f = get_filenames(self.new_root_path, fd)
                    for j, filename in enumerate(f):
                        file_j_df = pd.read_csv(filename, low_memory=False)
                        file_j_processed, _ = adp_filt(file_j_df, self.pose_chosen)
                        self.all_df.append(file_j_df)
                        self.new_data.append(file_j_processed)
                        self.filenames.append(filename)
                        self.folder.append(fd)
                    self.folders.append(fd)
            elif self.filetype == 'h5':
                try:
                    for i, fd in enumerate(self.new_directories):
                        f = get_filenamesh5(self.new_root_path, fd)
                        for j, filename in enumerate(f):
                            file_j_df = pd.read_hdf(filename, low_memory=False)
                            file_j_processed, _ = adp_filt_h5(file_j_df, self.pose_chosen)
                            self.all_df.append(file_j_df)
                            self.new_data.append(file_j_processed)
                            self.filenames.append(filename)
                            self.folder.append(fd)
                        self.folders.append(fd)
                except:
                    st.info('Detecting SLEAP .h5 files...')
                    for i, fd in enumerate(self.new_directories):
                        f = get_filenamesh5(self.new_root_path, fd)
                        for j, filename in enumerate(f):
                            file_j_df = h5py.File(filename, 'r')
                            file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
                            df = no_filt_sleap_h5(file_j_df, self.pose_chosen)
                            self.all_df.append(df)
                            self.new_data.append(file_j_processed)
                            self.filenames.append(filename)
                            self.folder.append(fd)
                        self.folders.append(fd)
            elif self.filetype == 'json':
                for i, fd in enumerate(self.new_directories):
                    f = get_filenamesjson(self.root_path, fd)
                    json2csv_multi(f)
                    filename = f[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
                    file_j_df = pd.read_csv(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')),
                                            low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                    self.all_df.append(file_j_df)
                    self.new_data.append(file_j_processed)
                    self.filenames.append(str.join('', (f[0].rpartition('/')[0], '/', filename, '.csv')))
                    self.folder.append(fd)
                    self.folders.append(fd)
            st.info('Extracting features and predicting labels... ')
            labels_fs = []
            bar = st.progress(0)
            for i in range(0, len(self.new_data)):
                feats_new = bsoid_extract([self.new_data[i]], self.new_framerate)
                labels = bsoid_predict(feats_new, self.clf)
                for m in range(0, len(labels)):
                    labels[m] = labels[m][::-1]
                labels_pad = -1 * np.ones([len(labels), len(max(labels, key=lambda x: len(x)))])
                for n, l in enumerate(labels):
                    labels_pad[n][0:len(l)] = l
                    labels_pad[n] = labels_pad[n][::-1]
                    if n > 0:
                        labels_pad[n][0:n] = labels_pad[n - 1][0:n]
                labels_fs.append(labels_pad.astype(int))
                bar.progress(round((i + 1) / len(self.new_data) * 100))
            st.info('Frameshift arrangement of predicted labels...')
            for k in range(0, len(labels_fs)):
                labels_fs2 = []
                for l in range(math.floor(self.new_framerate / 10)):
                    labels_fs2.append(labels_fs[k][l])
                self.new_predictions.append(np.array(labels_fs2).flatten('F'))
            st.info('Done frameshift-predicting a total of **{}** files.'.format(len(self.new_data)))
            for i in range(0, len(self.new_predictions)):
                filename_i = os.path.basename(self.filenames[i]).rpartition('.')[0]
                fs_labels_pad = np.pad(self.new_predictions[i], (0, len(self.all_df[i]) -
                                                                 len(self.new_predictions[i])), 'edge')
                df2 = pd.DataFrame(fs_labels_pad, columns={'B-SOiD labels'})
                frames = [df2, self.all_df[0]]
                xyfs_df = pd.concat(frames, axis=1)
                runlen_df, dur_stats, tm_array, tm_df, tm_norm = statistics.main(self.new_predictions[i],
                                                                                 len(np.unique(self.predictions)))
                try:
                    os.mkdir(str.join('', (self.new_root_path, self.folder[i], '/BSOID')))
                except FileExistsError:
                    pass
                if any('Labels tagged onto pose files' in o for o in self.options):
                    xyfs_df.to_csv(os.path.join(
                        str.join('', (self.new_root_path, self.folder[i], '/BSOID')),
                        str.join('', (self.new_prefix, 'labels_pose_', str(self.new_framerate),
                                      'Hz', filename_i, '.csv'))),
                        index=True, chunksize=10000, encoding='utf-8')
                    st.info('Saved Labels .csv in {}'.format(
                        str.join('', (self.new_root_path, self.folder[i], '/BSOID'))))
                if any('Group durations (in frames)' in o for o in self.options):
                    runlen_df.to_csv(os.path.join(
                        str.join('', (self.new_root_path, self.folder[i], '/BSOID')),
                        str.join('', (self.new_prefix, 'bout_lengths_', str(self.new_framerate),
                                      'Hz', filename_i, '.csv'))),
                        index=True, chunksize=10000, encoding='utf-8')
                    st.info('Saved Group durations .csv in {}'.format(
                        str.join('', (self.new_root_path, self.folder[i], '/BSOID'))))
                if any('Transition matrix' in o for o in self.options):
                    tm_df.to_csv(os.path.join(
                        str.join('', (self.new_root_path, self.folder[i], '/BSOID')),
                        str.join('', (self.new_prefix, 'transitions_mat_',
                                      str(self.new_framerate), 'Hz', filename_i, '.csv'))),
                        index=True, chunksize=10000, encoding='utf-8')
                    st.info('Saved transition matrix .csv in {}'.format(
                        str.join('', (self.new_root_path, self.folder[i], '/BSOID'))))
            with open(os.path.join(self.working_dir, str.join('', (self.new_prefix, '_predictions.sav'))), 'wb') as f:
                joblib.dump([self.folders, self.folder, self.filenames, self.new_data, self.new_predictions], f)
            st.balloons()
            st.markdown('**_CHECK POINT_**: Done predicting old/new files. Move on to '
                        '__Load up analysis app (please close current browser when new browser pops up)__.')

    def main(self):
        self.setup()
        self.predict()
