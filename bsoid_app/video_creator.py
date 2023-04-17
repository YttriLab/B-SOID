import base64

import ffmpeg
import h5py
import streamlit as st

from bsoid_app.bsoid_utilities.bsoid_classification import *
from bsoid_app.bsoid_utilities.likelihoodprocessing import *
from bsoid_app.bsoid_utilities.load_json import *
from bsoid_app.bsoid_utilities.videoprocessing import *


@st.cache(allow_output_mutation=True)
def selected_file(d_file):
    return d_file


@st.cache(allow_output_mutation=True)
def selected_vid(vid_file):
    return vid_file


class creator:

    def __init__(self, root_path, data_directories, processed_input_data,
                 pose_chosen, working_dir, prefix, framerate, clf, input_filenames):
        st.subheader('GENERATE VIDEOS SNIPPETS FOR INTERPRETATION')
        self.root_path = root_path
        self.data_directories = data_directories
        self.processed_input_data = processed_input_data
        self.pose_chosen = pose_chosen
        self.working_dir = working_dir
        self.prefix = prefix
        self.framerate = framerate
        self.clf = clf
        self.input_filenames = input_filenames
        self.file_directory = []
        self.d_file = []
        self.vid_dir = []
        self.vid_file = []
        self.frame_dir = []
        self.filetype = []
        self.width = []
        self.height = []
        self.bit_rate = []
        self.num_frames = []
        self.avg_frame_rate = []
        self.shortvid_dir = []
        self.min_frames = []
        self.number_examples = []
        self.out_fps = []
        self.file_j_processed = []

    def setup(self):
        if st.checkbox("Change __root directory__ to other than **{}**? Do this if you have another project "
                       "that benefits from built classifier.".format(self.root_path), False, 'vc'):
            self.root_path = st.text_input('Enter new __root directory__, e.g. /Users/projectY')
        self.file_directory = st.text_input(str.join('', ('Enter the __data containing sub-directory__'
                                                          ' within ', self.root_path)),
                                            self.data_directories[0])
        try:
            os.listdir(str.join('', (self.root_path, self.file_directory)))
            st.markdown('You have selected **{}** as your csv/h5/json data sub-directory.'.format(self.file_directory))
        except FileNotFoundError:
            st.error('No such directory')
        st.markdown('If your input was openpose **JSON(s)**, the app has converted into a SINGLE CSV for each folder. '
                    'Hence, the following will autodetect CSV as your filetype.')
        self.filetype = st.selectbox('What type of file?',
                                     ('csv', 'h5', 'json'),
                                     index=int([i for i, s in enumerate(['csv', 'h5', 'json'])
                                                if s in self.input_filenames[0].partition('.')[-1]][0]))
        if self.filetype == 'csv':
            d_file = st.selectbox('Select the csv file',
                                  sorted(os.listdir(str.join('', (self.root_path, self.file_directory)))))
            self.d_file = selected_file(d_file)
        elif self.filetype == 'h5':
            d_file = st.selectbox('Select the h5 file',
                                  sorted(os.listdir(str.join('', (self.root_path, self.file_directory)))))
            self.d_file = selected_file(d_file)
        elif self.filetype == 'json':
            d_files = get_filenamesjson(self.root_path, self.file_directory)
            fname = d_files[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
            if not os.path.isfile(str.join('', (d_files[0].rpartition('/')[0], '/', fname, '.csv'))):
                json2csv_multi(d_files)
            d_file = st.selectbox('Select the autocompiled csv file containing all jsons',
                                  sorted(os.listdir(str.join('', (self.root_path, self.file_directory)))))
            self.d_file = selected_file(d_file)
        self.vid_dir = st.text_input('Enter corresponding video directory (Absolute path):',
                                     str.join('', (self.root_path, self.data_directories[0])))
        try:
            os.listdir(self.vid_dir)
            st.markdown(
                'You have selected **{}** as your video directory.'.format(self.vid_dir))
        except FileNotFoundError:
            st.error('No such directory')
        vid_file = st.selectbox('Select the video (.mp4 or .avi)', sorted(os.listdir(self.vid_dir)))
        self.vid_file = selected_vid(vid_file)
        if self.filetype == 'csv' or self.filetype == 'h5':
            st.markdown('You have selected **{}** matching **{}**.'.format(self.vid_file, self.d_file))
            csvname = os.path.basename(self.d_file).rpartition('.')[0]
        else:
            st.markdown(
                'You have selected **{}** matching **{}** json directory.'.format(self.vid_file, self.file_directory))
            csvname = os.path.basename(self.file_directory)
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/pngs')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/pngs', '/', csvname)))
        except FileExistsError:
            pass
        self.frame_dir = str.join('', (self.root_path, self.file_directory, '/pngs', '/', csvname))
        st.markdown('Created {} as your **video frames** directory.'.format(self.frame_dir, self.vid_file))
        probe = ffmpeg.probe(os.path.join(self.vid_dir, self.vid_file))
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.num_frames = int(video_info['nb_frames'])
        self.bit_rate = int(video_info['bit_rate'])
        self.avg_frame_rate = round(
            int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/mp4s')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (self.root_path, self.file_directory, '/mp4s', '/', csvname)))
        except FileExistsError:
            pass
        self.shortvid_dir = str.join('', (self.root_path, self.file_directory, '/mp4s', '/', csvname))
        st.markdown('Created {} as your **behavioral snippets** directory.'.format(self.shortvid_dir, self.vid_file))
        min_time = st.number_input('Enter minimum time for bout in ms:', value=200)
        self.min_frames = round(float(min_time) * 0.001 * float(self.framerate))
        st.markdown('Entered **{} ms** as minimum duration per bout, '
                    'which is equivalent to **{} frames**.'.format(min_time, self.min_frames))
        self.number_examples = st.slider('Select number of non-repeated examples', 1, 20, 5)
        st.markdown(
            'Your will obtain a maximum of **{}** non-repeated output examples per group.'.format(self.number_examples))
        playback_speed = st.number_input('Enter playback speed:', value=0.75)
        self.out_fps = int(float(playback_speed) * float(self.framerate))
        st.markdown('Playback at **{} x speed** (rounded to {} FPS).'.format(playback_speed, self.out_fps))

    def frame_extraction(self):
        if st.button('Start frame extraction for {} frames '
                     'at {} frames per second'.format(self.num_frames, self.avg_frame_rate)):
            st.info('Extracting frames from the video... ')
            try:
                (ffmpeg.input(os.path.join(self.vid_dir, self.vid_file))
                 .filter('fps', fps=self.avg_frame_rate)
                 .output(str.join('', (self.frame_dir, '/frame%01d.png')), video_bitrate=self.bit_rate,
                         s=str.join('', (str(int(self.width * 0.5)), 'x', str(int(self.height * 0.5)))),
                         sws_flags='bilinear', start_number=0)
                 .run(capture_stdout=True, capture_stderr=True))
                st.info('Done extracting **{}** frames from video **{}**.'.format(self.num_frames, self.vid_file))
            except ffmpeg.Error as e:
                st.error('stdout:', e.stdout.decode('utf8'))
                st.error('stderr:', e.stderr.decode('utf8'))
            st.info('Done extracting {} frames from {}'.format(self.num_frames, self.vid_file))

    def create_videos(self):
        radio = st.radio(label='Have you extracted frames?', options=["Yes", "No"])
        if radio == 'Yes':
            if st.checkbox('Clear old videos? Uncheck after check to prevent from auto-clearing', False, key='vr'):
                try:
                    for file_name in glob.glob(self.shortvid_dir + "/*"):
                        os.remove(file_name)
                except:
                    pass
            if st.button("Predict labels and create example videos"):
                if self.filetype == 'csv' or self.filetype == 'json':
                    file_j_df = pd.read_csv(
                        os.path.join(str.join('', (self.root_path, self.file_directory, '/', self.d_file))),
                        low_memory=False)
                    file_j_processed, p_sub_threshold = adp_filt(file_j_df, self.pose_chosen)
                elif self.filetype == 'h5':
                    try:
                        file_j_df = pd.read_hdf(
                            os.path.join(str.join('', (self.root_path, self.file_directory, '/', self.d_file))),
                            low_memory=False)
                        file_j_processed, p_sub_threshold = adp_filt_h5(file_j_df, self.pose_chosen)
                    except:
                        st.info('Detecting a SLEAP .h5 file...')
                        file_j_df = h5py.File(
                            os.path.join(str.join('', (self.root_path, self.file_directory, '/', self.d_file))), 'r')
                        file_j_processed, p_sub_threshold = adp_filt_sleap_h5(file_j_df, self.pose_chosen)
                self.file_j_processed = [file_j_processed]
                labels_fs = []
                fs_labels = []
                st.info('Predicting labels... ')
                for i in range(0, len(self.file_j_processed)):
                    feats_new = bsoid_extract([self.file_j_processed[i]], self.framerate)
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
                st.info('Frameshifted arrangement of labels... ')
                for k in range(0, len(labels_fs)):
                    labels_fs2 = []
                    for l in range(math.floor(self.framerate / 10)):
                        labels_fs2.append(labels_fs[k][l])
                    fs_labels.append(np.array(labels_fs2).flatten('F'))
                st.info('Done frameshift-predicting **{}**.'.format(self.d_file))
                create_labeled_vid(fs_labels[0], int(self.min_frames), int(self.number_examples), int(self.out_fps),
                                   self.frame_dir, self.shortvid_dir)
                st.balloons()
                st.markdown('**_CHECK POINT_**: Done generating video snippets. Move on to '
                            '__Predict old/new files using a model__.')
        elif radio == 'No':
            self.frame_extraction()

    def show_snippets(self):
        video_bytes = []
        grp_names = []
        files = []
        for file in os.listdir(self.shortvid_dir):
            files.append(file)
        sort_nicely(files)
        st.info('Creating gifs from mp4s...')
        for file in files:
            if file.endswith('0.mp4'):
                try:
                    example_vid_file = open(os.path.join(
                        str.join('', (self.shortvid_dir, '/', file.partition('.')[0], '.gif'))), 'rb')
                except FileNotFoundError:
                    convert2gif(str.join('', (self.shortvid_dir, '/', file)), TargetFormat.GIF)
                    example_vid_file = open(os.path.join(
                        str.join('', (self.shortvid_dir, '/', file.partition('.')[0], '.gif'))), 'rb')
                contents = example_vid_file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                video_bytes.append(data_url)
                grp_names.append('{}'.format(file.partition('.')[0]))
        col = [None] * 3
        col[0], col[1], col[2] = st.columns([1, 1, 1])
        for i in range(0, len(video_bytes) + 3, 3):
            try:
                col[0].markdown(
                    f'<div class="container">'
                    f'<img src="data:image/gif;base64,{video_bytes[i]}" alt="" width="300" height="300">'
                    f'<div class="bottom-left">{grp_names[i]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                col[1].markdown(
                    f'<div class="container">'
                    f'<img src="data:image/gif;base64,{video_bytes[i + 1]}" alt="" width="300" height="300">'
                    f'<div class="bottom-left">{grp_names[i + 1]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                col[2].markdown(
                    f'<div class="container">'
                    f'<img src="data:image/gif;base64,{video_bytes[i + 2]}" alt="" width="300" height="300">'
                    f'<div class="bottom-left">{grp_names[i + 2]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            except IndexError:
                pass

    def main(self):
        self.setup()
        self.create_videos()
        if st.checkbox("Show a collage of example group? "
                       "This could take some time for gifs conversions.".format(self.shortvid_dir), False, key='vs'):
            self.show_snippets()

