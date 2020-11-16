import h5py
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis_subroutines.analysis_scripts.trajectory_plot import *


class trajectory:

    def __init__(self, working_dir, prefix, framerate, filenames, new_data, new_predictions):
        st.subheader('LIMB TRAJECTORIES (PAPER **FIGURE 2D/G**)')
        self.working_dir = working_dir
        self.prefix = prefix
        self.framerate = framerate
        self.filenames = filenames
        self.new_data = new_data
        self.new_predictions = new_predictions
        self.animal_index = int(st.number_input('Which session? '
                                                'You have a total of {} sessions'.format(len(self.new_predictions)),
                                                min_value=1, max_value=len(self.new_predictions), value=1)) - 1
        self.c = []
        self.pose_chosen = []
        file_type = [s for i, s in enumerate(['csv', 'h5', 'json']) if s in self.filenames[0].partition('.')[-1]][0]
        if file_type == 'csv':
            file0_df = pd.read_csv(self.filenames[0], low_memory=False)
            file0_array = np.array(file0_df, dtype=object)
            p = st.multiselect('Select body parts for trajectory:', [*file0_array[0, 1:-1:3]],
                               [*file0_array[0, 1:-1:3]])
            for b in p:
                index = [i for i, s in enumerate(file0_array[0, 1:-1:3]) if b in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
            self.pose_chosen.sort()
        elif file_type == 'h5':
            try:
                file0_df = pd.read_hdf(self.filenames[0], low_memory=False)
                p = st.multiselect('Identified __pose__ to include:',
                                   [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])],
                                   [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])])
                for b in p:
                    index = [i for i, s in enumerate(np.array(file0_df.columns.get_level_values(1))) if b in s]
                    if not index in self.pose_chosen:
                        self.pose_chosen += index
                self.pose_chosen.sort()
            except:
                st.info('Detecting SLEAP h5 files...')
                file0_df = h5py.File(self.filenames[0], 'r')
                p = st.multiselect('Identified __pose__ to include:',
                                   [*np.array(file0_df['node_names'][:])],
                                   [*np.array(file0_df['node_names'][:])])
                for b in p:
                    index = [i for i, s in enumerate(np.array(file0_df['node_names'][:])) if b in s]
                    if not index in self.pose_chosen:
                        self.pose_chosen += index
                self.pose_chosen.sort()
        start_min = int(st.number_input('From minute:',
                                        min_value=0,
                                        max_value=int((len(self.new_data[self.animal_index])) /
                                                      (60 * self.framerate)), value=20))
        start_sec = st.number_input('and second:', min_value=0.0, max_value=59.9, value=42.0)
        stop_min = int(st.number_input('till minute:',
                                       min_value=0,
                                       max_value=int((len(self.new_data[self.animal_index])) /
                                                     (60 * self.framerate)), value=20))
        stop_sec = st.number_input('till second:', min_value=0.0, max_value=59.9, value=44.0)
        start = int((start_min * 60 + start_sec) * self.framerate) - 1
        stop = int((stop_min * 60 + stop_sec) * self.framerate)
        self.time_range = [start, stop]
        order_top = st.multiselect('Which should be grouped to the top?', p, p)
        self.order1 = []
        for o in order_top:
            index = [i for i, s in enumerate(p) if o in s]
            if not index in self.order1:
                self.order1 += index
        order_bottom = st.multiselect('Which should be grouped to the bottom?', p, p)
        self.order2 = []
        for o in order_bottom:
            index = [i for i, s in enumerate(p) if o in s]
            if not index in self.order2:
                self.order2 += index
        color1 = st.selectbox('Choose color for first group', list(mcolors.CSS4_COLORS.keys()), index=16)
        color2 = st.selectbox('Choose color for second group', list(mcolors.CSS4_COLORS.keys()), index=22)
        self.c = [color1, color2]

    def plot(self):
        if st.checkbox('Show trajectory', False, key='tp'):
            labels, limbs, soft_assigns = limb_trajectory(self.working_dir, self.prefix,
                                                          self.animal_index, self.pose_chosen, self.time_range)
            fig, ax1, ax2 = plot_trajectory(limbs, labels, soft_assigns, self.time_range,
                                            self.order1, self.order2, self.c,
                                            fig_size=(5, 3), save=False)
            fig.suptitle('Trajectory visual')
            ax1.set_ylabel('$\Delta$ pixels')
            ax2.set_ylabel('$\Delta$ pixels')
            ax2.set_xlabel('Frame number')
            st.pyplot(fig)
            fig_format = str(st.selectbox('What file type?',
                                          list(plt.gcf().canvas.get_supported_filetypes().keys()), index=5))
            outpath = str.join('', (st.text_input('Where would you like to save it?'), '/'))
            if st.button('Save in {}?'.format(outpath)):
                plot_trajectory(limbs, labels, self.time_range, self.order1, self.order2,
                                self.c, (8.5, 16), fig_format, outpath, True)

    def main(self):
        self.plot()
