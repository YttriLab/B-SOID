import matplotlib.colors as mcolors
import streamlit as st

from analysis_subroutines.analysis_scripts.kfold_accuracy import *
from analysis_subroutines.analysis_utilities.load_data import load_sav
from analysis_subroutines.analysis_utilities.save_data import results
from analysis_subroutines.analysis_utilities.visuals import *


class performance:

    def __init__(self, working_dir, prefix, soft_assignments):
        st.subheader('K-FOLD ACCURACY (PAPER **FIGURE 2C**)')
        self.working_dir = working_dir
        self.prefix = prefix
        self.soft_assignments = soft_assignments
        self.k = int(st.number_input('How many folds cross-validation?', min_value=2, max_value=20, value=10))
        self.var_name = 'accuracy_kf_raw'
        self.var_ordered_name = 'accuracy_kf_ordered'
        self.order_class = st.multiselect('Order as follows:', list(np.unique(self.soft_assignments)),
                                          list(np.unique(self.soft_assignments)))
        self.accuracy_data = []
        self.accuracy_ordered = []

    def cross_validate(self):
        if st.button('Start K-fold cross-validation.', False):
            self.accuracy_data = generate_kfold(self.working_dir, self.prefix, self.k)
            results_ = results(self.working_dir, self.prefix)
            results_.save_sav([self.accuracy_data, self.k], self.var_name)
            self.accuracy_ordered = reorganize_group_order(self.accuracy_data, self.order_class)
            results_ = results(self.working_dir, self.prefix)
            results_.save_sav([self.accuracy_ordered, self.k], self.var_ordered_name)

    def load_performance(self):
        self.accuracy_ordered, self.k = load_sav(self.working_dir, self.prefix, self.var_ordered_name)

    def show_accuracy_plot(self):
        c = [list(mcolors.CSS4_COLORS.keys())[122]] * len(np.unique(self.soft_assignments))
        fig, ax = plot_accuracy_boxplot(None, self.accuracy_ordered, c, (5, 3), save=False)
        fig.suptitle('{}-fold group accuracy'.format(self.k))
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Group number')
        col1, col2 = st.beta_columns([2, 2])
        radio = st.radio(label='Change colors?', options=["Yes", "No"], index=1)
        if radio == 'No':
            col1.pyplot(fig)
            fig_format = str(st.selectbox('What file type?',
                                          list(plt.gcf().canvas.get_supported_filetypes().keys()), index=5))
            out_path = str.join('', (st.text_input('Where would you like to save it?'), '/'))
            if st.button('Save in {}?'.format(out_path)):
                plot_accuracy_boxplot('Randomforests', self.accuracy_ordered, c,
                                      (8.5, 16), fig_format, out_path, save=True)
        elif radio == 'Yes':
            c = []
            for i in range(len(np.unique(self.soft_assignments))):
                color = st.selectbox('Choose color for ORDERED GROUP {}'.format(i),
                                     list(mcolors.CSS4_COLORS.keys()), index=122)
                c.append(color)
            fig2, ax2 = plot_accuracy_boxplot(None, self.accuracy_ordered, c, (5, 3), save=False)
            fig2.suptitle('{}-fold group accuracy'.format(self.k))
            ax2.set_xlabel('Accuracy')
            ax2.set_ylabel('Group number')
            try:
                col1.pyplot(fig2)
            except ValueError:
                st.error('Try another color, this color is not supported :/')
            fig_format = str(st.selectbox('What file type?',
                                          list(plt.gcf().canvas.get_supported_filetypes().keys()), index=5))
            out_path = str.join('', (st.text_input('Where would you like to save it?'), '/'))
            if st.button('Save in {}?'.format(out_path)):
                plot_accuracy_boxplot('Randomforests', self.accuracy_ordered, c,
                                      (8.5, 16), fig_format, out_path, save=True)

    def main(self):
        try:
            self.load_performance()
        except:
            self.cross_validate()
            self.load_performance()
        self.show_accuracy_plot()
