import h5py
import matplotlib.colors as mcolors
import pandas as pd
import streamlit as st

from analysis_subroutines.analysis_scripts.extract_kinematics import *
from analysis_subroutines.analysis_utilities.visuals import *
from analysis_subroutines.analysis_utilities.save_data import results
from analysis_subroutines.analysis_utilities.load_data import load_sav


class kinematics:

    def __init__(self, working_dir, prefix, framerate, soft_assignments, filenames):
        st.subheader('(BETA) KINEMATICS (PAPER **FIGURE 6B/D**)')
        self.working_dir = working_dir
        self.prefix = prefix
        self.framerate = framerate
        self.soft_assignments = soft_assignments
        self.filenames = filenames
        self.group_num = int(st.number_input('Which behavioral group? You have a total of {} groups '
                                             'starting from 0 to {}'.format(len(np.unique(self.soft_assignments)),
                                                                            len(np.unique(self.soft_assignments)) - 1),
                                             min_value=0, max_value=len(np.unique(self.soft_assignments)), value=0))
        self.c = []
        self.bps_exp1_bout_disp = []
        self.bps_exp2_bout_disp = []
        self.bps_exp1_bout_peak_speed = []
        self.bps_exp2_bout_peak_speed = []
        self.bps_exp1_bout_dur = []
        self.bps_exp2_bout_dur = []
        self.vid_outpath = []
        self.pose_chosen = []
        file_type = [s for i, s in enumerate(['csv', 'h5', 'json']) if s in self.filenames[0].partition('.')[-1]][0]
        if file_type == 'csv':
            file0_df = pd.read_csv(self.filenames[0], low_memory=False)
            file0_array = np.array(file0_df, dtype=object)
            self.p = st.multiselect('Select body parts for kinematics:', [*file0_array[0, 1:-1:3]],
                                    [*file0_array[0, 1:-1:3]])
            for b in self.p:
                index = [i for i, s in enumerate(file0_array[0, 1:-1:3]) if b in s]
                if not index in self.pose_chosen:
                    self.pose_chosen += index
        elif file_type == 'h5':
            try:
                file0_df = pd.read_hdf(self.filenames[0], low_memory=False)
                self.p = st.multiselect('Identified __pose__ to include:',
                                        [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])],
                                        [*np.array(file0_df.columns.get_level_values(1)[1:-1:3])])
                for b in self.p:
                    index = [i for i, s in enumerate(np.array(file0_df.columns.get_level_values(1)[:])) if b in s]
                    if not index in self.pose_chosen:
                        self.pose_chosen += index
            except:
                file0_df = h5py.File(self.filenames[0], 'r')
                self.p = st.multiselect('Identified __pose__ to include:',
                                        [*np.array(file0_df['node_names'][:])],
                                        [*np.array(file0_df['node_names'][:])])
                for b in self.p:
                    index = [i for i, s in enumerate(np.array(file0_df['node_names'][:])) if b in s]
                    if not index in self.pose_chosen:
                        self.pose_chosen += index
        self.pose_chosen.sort()
        fname_partition = [self.filenames[i].rpartition('/')[-1] for i in range(len(self.filenames))]
        order_ctrl = st.multiselect('Which sessions to be grouped together as CONTROL?',
                                    fname_partition, fname_partition)
        self.control = []
        for o in order_ctrl:
            index = [i for i, s in enumerate(self.filenames) if o in s]
            if not index in self.control:
                self.control += index
        color1 = st.selectbox('Choose color for CONTROL', list(mcolors.CSS4_COLORS.keys()), index=41)
        order_expt = st.multiselect('Which sessions to be grouped together as EXPERIMENTAL?',
                                    fname_partition, fname_partition)
        self.experimental = []
        for o in order_expt:
            index = [i for i, s in enumerate(filenames) if o in s]
            if not index in self.experimental:
                self.experimental += index
        self.conditions = [self.control, self.experimental]
        color2 = st.selectbox('Choose color for EXPERIMENTAL', list(mcolors.CSS4_COLORS.keys()), index=120)
        self.c = [color1, color2]
        self.variable_name = st.text_input('Give it a variable name to save:')

    def find_peaks(self):
        if st.button('Start analyzing kinematics?'):
            [_, _, all_bouts_disp, all_bouts_peak_speed, all_bouts_dur, self.vid_outpath] = \
                get_kinematics(self.working_dir, self.prefix, self.conditions, self.group_num,
                               self.pose_chosen, self.framerate)
            [self.bps_exp1_bout_disp, self.bps_exp2_bout_disp, self.bps_exp1_bout_peak_speed,
             self.bps_exp2_bout_peak_speed, self.bps_exp1_bout_dur, self.bps_exp2_bout_dur] = \
                group_kinematics(all_bouts_disp, all_bouts_peak_speed, all_bouts_dur, self.conditions)
            results_ = results(self.working_dir, self.prefix)
            results_.save_sav(
                [self.bps_exp1_bout_disp, self.bps_exp2_bout_disp, self.bps_exp1_bout_peak_speed,
                 self.bps_exp2_bout_peak_speed, self.bps_exp1_bout_dur, self.bps_exp2_bout_dur, self.p,
                 self.pose_chosen,
                 self.conditions, self.vid_outpath], self.variable_name)
            st.info('Done analyzing kinematics. Click "R" for plots.')

    def plot(self, save, out_path, fig_format):
        for pose in range(len(self.p)):
            f, ax = plt.subplots(1, 3)
            f.suptitle('Kinematics CDF for {}'.format(self.variable_name))
            try:
                fig1, ax1 = plot_kinematics_cdf(ax.flatten()[0], 'distance', self.variable_name,
                                                [self.bps_exp1_bout_disp[pose], self.bps_exp2_bout_disp[pose]],
                                                self.c, 50, 3, 1, fig_size=(5, 3), save=False)
                ax1.title.set_text('Distance')
                ax1.set_xlabel('Dist. ($\Delta$ pix)'.format(self.p[pose]))
                ax1.set_ylabel('Cumulative probability')
            except:
                pass
            try:
                fig2, ax2 = plot_kinematics_cdf(ax.flatten()[1], 'speed', self.variable_name,
                                                [self.bps_exp1_bout_peak_speed[pose],
                                                 self.bps_exp2_bout_peak_speed[pose]],
                                                self.c, 50, 3, 1, fig_size=(5, 3), save=False)
                ax2.title.set_text('Peak speed')
                ax2.set_xlabel('{} speed (pix/frm)'.format(self.p[pose]))
                ax2.yaxis.set_ticklabels([])
            except:
                pass
            try:
                fig3, ax3 = plot_kinematics_cdf(ax.flatten()[2], 'duration', self.variable_name,
                                                [self.bps_exp1_bout_dur[pose] / self.framerate * 1000,
                                                 self.bps_exp2_bout_dur[pose] / self.framerate * 1000],
                                                self.c, 50, 3, 1, fig_size=(5, 3), save=False)
                ax3.title.set_text('Bout duration')
                ax3.set_xlabel('Duration (ms)'.format(self.p[pose]))
                ax3.yaxis.set_ticklabels([])
            except:
                pass
            st.pyplot(f)
            if save:
                try:
                    f.savefig(str.join('', (out_path, '/{}_kin_{}_cdf.'.format(self.p[pose], self.variable_name), '.',
                                            fig_format)), dpi=300, format=fig_format, transparent=False)
                except RuntimeError:
                    st.error('Could not save in this format, find another one (jpeg/png/svg)?')

    def main(self):
        try:
            [self.bps_exp1_bout_disp, self.bps_exp2_bout_disp, self.bps_exp1_bout_peak_speed,
             self.bps_exp2_bout_peak_speed, self.bps_exp1_bout_dur, self.bps_exp2_bout_dur, self.p, self.pose_chosen,
             self.conditions, self.vid_outpath] = load_sav(self.working_dir, self.prefix, self.variable_name)
            st.markdown('**Peak speed** computed with ***instantaneous values of peaks***. '
                        '**Distance** computed using the ***area under the curve from start to end '
                        '(colored)*** of peak. '
                        '**Bout duration** computed using ***number of consecutive frames*** in B-SOID defined bouts. '
                        'The pose trajectory algorithm performance can be visualized above '
                        '(checkbox, 50% random samples).')
            if st.checkbox('Redo?', key='r'):
                self.find_peaks()
            ftype_out = st.selectbox('What file type?',
                                     list(plt.gcf().canvas.get_supported_filetypes().keys()), index=5)
            out_path = str.join('', (st.text_input('Where would you like to save it?'), '/'))
            save = st.checkbox('Save in {}?'.format(out_path), False, key='sa')
            if save:
                self.plot(save=True, out_path=out_path, fig_format=ftype_out)
            else:
                self.plot(save=False, out_path=out_path, fig_format=ftype_out)
            if st.checkbox('Show peak finding algorithm performance?', False):
                example_vid_file = open(os.path.join(str.join('', (self.vid_outpath,
                                                                   '/kinematics_subsample_examples.mp4'))), 'rb')
                st.markdown('You have selected to view examples from {}.'.format(self.vid_outpath))
                video_bytes = example_vid_file.read()
                st.video(video_bytes)
        except:
            self.find_peaks()

