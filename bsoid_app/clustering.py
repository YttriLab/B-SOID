import os

import hdbscan
import joblib
import numpy as np
import streamlit as st
import randfacts

from bsoid_app.config import *
from bsoid_app.bsoid_utilities import visuals
from bsoid_app.bsoid_utilities.load_workspace import load_clusters
from streamlit import caching


class cluster:

    def __init__(self, working_dir, prefix, sampled_embeddings):
        st.subheader('IDENTIFY AND TWEAK NUMBER OF CLUSTERS.')
        self.working_dir = working_dir
        self.prefix = prefix
        self.sampled_embeddings = sampled_embeddings
        self.cluster_range = []
        self.min_cluster_size = []
        self.assignments = []
        self.assign_prob = []
        self.soft_assignments = []

    def hierarchy(self):
        if st.button("__Identify Clusters__"):
            funfacts = randfacts.getFact()
            st.info(str.join('', ('Identifying... Here is a random fact: ', funfacts)))
            max_num_clusters = -np.infty
            num_clusters = []
            self.min_cluster_size = np.linspace(self.cluster_range[0], self.cluster_range[1], 25)

            for min_c in self.min_cluster_size:
                learned_hierarchy = hdbscan.HDBSCAN(
                    prediction_data=True, min_cluster_size=int(round(min_c * 0.01 * self.sampled_embeddings.shape[0])),
                    **HDBSCAN_PARAMS).fit(self.sampled_embeddings)
                num_clusters.append(len(np.unique(learned_hierarchy.labels_)))
                if num_clusters[-1] > max_num_clusters:
                    max_num_clusters = num_clusters[-1]
                    retained_hierarchy = learned_hierarchy
            self.assignments = retained_hierarchy.labels_
            self.assign_prob = hdbscan.all_points_membership_vectors(retained_hierarchy)
            self.soft_assignments = np.argmax(self.assign_prob, axis=1)
            st.info('Done assigning labels for **{}** instances ({} minutes) '
                    'in **{}** D space'.format(self.assignments.shape,
                                               round(self.assignments.shape[0] / 600),
                                               self.sampled_embeddings.shape[1]))
            st.balloons()

    def show_classes(self):
        st.write('Showing {}% data that were confidently assigned.'
                 ''.format(round(self.assignments[self.assignments >= 0].shape[0] /
                                 self.assignments.shape[0] * 100)))
        fig1, plt1 = visuals.plot_classes(self.sampled_embeddings[self.assignments >= 0],
                                          self.assignments[self.assignments >= 0])
        plt1.suptitle('HDBSCAN assignment')
        col1, col2 = st.columns([2, 2])
        col1.pyplot(fig1)

    def slider(self, min_=0.5, max_=1.0):
        st.markdown('The following slider allows you to tweak number of groups based on minimum size requirements.')
        st.text('')
        self.cluster_range = st.slider('Select range of __minimum cluster size__ in %', 0.01, 5.0, (min_, max_))
        st.markdown('Your minimum cluster size ranges between **{}%** and **{}%**, '
                    'which is equivalent to roughly {} seconds for the '
                    'smallest cluster.'.format(self.cluster_range[0], self.cluster_range[1],
                                               round(self.cluster_range[0] * 0.001 * self.sampled_embeddings.shape[0])))

    def save(self):
        save_ = st.radio('Autosave the clustering as you go? This will overwrite the previous saved clustering.',
                         options=['Yes', 'No'], index=0)
        if save_ == 'Yes':
            with open(os.path.join(self.working_dir, str.join('', (self.prefix, '_clusters.sav'))), 'wb') as f:
                joblib.dump([self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments], f)
        st.text('')
        st.text('')

    def main(self):
        try:
            caching.clear_cache()
            [self.min_cluster_size, self.assignments, self.assign_prob, self.soft_assignments] = \
                load_clusters(self.working_dir, self.prefix)
            st.markdown(
                '**_CHECK POINT_**: Done assigning labels for **{}** instances in **{}** D space. Move on to __create '
                'a model__.'.format(self.assignments.shape, self.sampled_embeddings.shape[1]))
            st.markdown('Your last saved run range was __{}%__ to __{}%__'.format(self.min_cluster_size[0],
                                                                                  self.min_cluster_size[-1]))
            if st.checkbox('Redo?', False, key='cr'):
                caching.clear_cache()
                self.slider(min_=float(self.min_cluster_size[0]), max_=float(self.min_cluster_size[-1]))
                self.hierarchy()
                self.save()
            if st.checkbox("Show first 3D UMAP enhanced clustering plot?", True, key='cs'):
                self.show_classes()
        except (AttributeError, FileNotFoundError) as e:
            self.slider()
            self.hierarchy()
            self.save()
            if st.checkbox("Show first 3D UMAP enhanced clustering plot?", True, key='cs'):
                self.show_classes()



