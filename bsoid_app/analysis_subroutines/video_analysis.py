import ffmpeg
import streamlit as st

from analysis_subroutines.analysis_utilities.visuals import *
from analysis_subroutines.analysis_scripts.umap_clustering_plot import plot_enhanced_umap


class bsoid_video:

    def __init__(self, working_dir, prefix, features, sampled_features,
                 sampled_embeddings, soft_assignments, framerate, filenames, new_data):
        st.subheader('SYNCHRONIZED B-SOID VIDEO (PAPER **SUPP. VIDEO 1**)')
        st.markdown('Cluster plot from training, we will see how well purely clustering maps onto behaviors by '
                    'synchronizing the video next to it')
        self.working_dir = working_dir
        self.prefix = prefix
        self.features = features
        self.sampled_features = sampled_features
        self.sampled_embeddings = sampled_embeddings
        self.soft_assignments = soft_assignments
        self.framerate = framerate
        self.filenames = filenames
        self.new_data = new_data
        sim_array = np.arange(0, self.features.shape[1])
        np.random.seed(0)
        shuffled_idx = np.random.choice(sim_array, self.sampled_features.shape[0], replace=False)
        ordered_ind = np.argsort(shuffled_idx)
        self.ordered_embeds = self.sampled_embeddings[ordered_ind, :]
        self.ordered_assigns = self.soft_assignments[ordered_ind]
        fig, ax = plot_enhanced_umap(self.working_dir, self.prefix, fig_size=(5, 3), save=False)
        col1, col2 = st.beta_columns([2, 2])
        col1.pyplot(fig)
        self.vid_path = st.text_input('Enter corresponding video directory (Absolute path):')
        try:
            os.listdir(self.vid_path)
            st.markdown('You have selected **{}** as your video directory.'.format(self.vid_path))
        except FileNotFoundError:
            st.error('No such directory')
        self.vid_name = st.selectbox('Select the video (.mp4 or .avi)', sorted(os.listdir(self.vid_path)))
        f_partition = [self.filenames[i].rpartition('/')[-1] for i in range(len(self.filenames))]
        file4vid = st.selectbox('Which file corresponds to the video?',
                                f_partition, index=0)
        f_index = f_partition.index(file4vid)
        probe = ffmpeg.probe(os.path.join(self.vid_path, self.vid_name))
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        self.width = int(video_info['width'])
        self.height = int(video_info['height'])
        self.mov_st_min = int(st.number_input('From minute:',
                                              min_value=0,
                                              max_value=int((len(self.new_data[f_index])) / (60 * self.framerate)),
                                              value=0))
        self.mov_st_sec = st.number_input('and second:', min_value=0.0, max_value=59.9, value=0.0)
        self.mov_sp_min = int(st.number_input('till minute:',
                                              min_value=0,
                                              max_value=int((len(self.new_data[f_index])) / (60 * self.framerate)),
                                              value=0))
        self.mov_sp_sec = st.number_input('till second:', min_value=0.0, max_value=59.9, value=1.0)
        self.mov_start = np.sum([len(self.new_data[j]) for j in np.arange(0, f_index)]) + \
                         int((self.mov_st_min * 60 + self.mov_st_sec) * self.framerate) - 1
        self.mov_stop = np.sum([len(self.new_data[j]) for j in np.arange(0, f_index)]) + \
                        int((self.mov_sp_min * 60 + self.mov_sp_sec) * self.framerate)
        self.mov_range = [round(self.mov_start / (self.framerate / 10)), round(self.mov_stop / (self.framerate / 10))]
        try:
            os.mkdir(str.join('', (self.working_dir, '/bsoid_videos')))
        except FileExistsError:
            pass
        try:
            os.mkdir(str.join('', (self.working_dir, '/bsoid_videos/session{}'.format(f_index))))
        except FileExistsError:
            pass
        self.working_dir = str.join('', (self.working_dir, '/bsoid_videos/session{}'.format(f_index)))

    def generate(self):
        if st.button('Generate synchronized B-SOiD video?'):
            try:
                umap_scatter(self.ordered_embeds, self.ordered_assigns, self.mov_range,
                             self.working_dir, self.width, self.height)
                trim_video(self.vid_path, self.vid_name, self.mov_range,
                           self.mov_st_min, self.mov_st_sec, self.mov_sp_min, self.mov_sp_sec, self.working_dir)
                video_umap(self.working_dir, self.mov_range)
            except IndexError:
                st.error('Range out of bounds!')
        if st.checkbox('Show left-right synchronized B-SOiD video (from {})?'.format(self.working_dir)):
            bsoid_vid_leftright = \
                open(os.path.join(str.join('', (self.working_dir,
                                                '/sync_leftright_video2umap{}_{}.mp4'.format(*self.mov_range)))), 'rb')
            st.markdown('You have selected to view synchronized video from {}.'.format(self.working_dir))
            bsoid_leftright_bytes = bsoid_vid_leftright.read()
            st.video(bsoid_leftright_bytes)
        st.markdown('After we visualize how well clusters map onto behaviors, we can utilize '
                    'a machine learning classifier to make more generalized prediction. '
                    'We can look at how well the machine learns the mapping in the **k-fold accuracy** module.')

    def main(self):
        self.generate()

