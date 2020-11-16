import itertools
import os

import streamlit as st

from bsoid_app.bsoid_utilities.statistics import *


class export:

    def __init__(self, working_dir, prefix, sampled_features, assignments, assign_prob, soft_assignments):
        st.subheader('WHAT DID B-SOID LEARN?')
        self.options = st.multiselect('What do you want to know?',
                                      ['List of pose-relationship features',
                                       'Mapping between features and assignment',
                                       'Assignment probabilities'],
                                      ['Mapping between features and assignment'])
        try:
            self.working_dir = working_dir
            self.prefix = prefix
            self.sampled_features = sampled_features
            self.assignments = assignments
            self.assign_prob = assign_prob
            self.soft_assignments = soft_assignments
        except FileNotFoundError:
            st.error('Cannot find training, please complete all the necessary steps')

    def save_csv(self):
        if st.button('Generate training features and probabilities'):
            if any('List of pose-relationship features' in o for o in self.options):
                feats_range, feats_median, feats_pct, edges = feat_dist(self.sampled_features)
                feats_med_df = pd.DataFrame(feats_median, columns=['median'])
                feats_pcts_df = pd.DataFrame(feats_pct)
                feats_edge_df = pd.DataFrame(edges)
                feats_edge_df.columns = pd.MultiIndex.from_product([['histogram edge'], feats_edge_df.columns])
                feats_pcts_df.columns = pd.MultiIndex.from_product([['histogram prob for edge'], feats_pcts_df.columns])
                f_dist_data = pd.concat((feats_med_df, feats_edge_df, feats_pcts_df), axis=1)
                f_dist_data.index.name = 'Pose_relationships'
                f_dist_data.to_csv(
                    (os.path.join(self.working_dir, str.join('', (self.prefix, '_pose_relationships.csv')))),
                    index=True, chunksize=10000, encoding='utf-8')
            if any('Mapping between features and assignment' in o for o in self.options):
                feature_type1_name = []
                feature_type2_name = []
                feature_type3_name = []
                for i, j in itertools.combinations(range(0, int(np.sqrt(self.sampled_features.shape[1]))), 2):
                    feature_type1_name.append(['Pose ', i + 1, j + 1, 'delta pixels'])
                    feature_type2_name.append(['Pose vector ', i + 1, j + 1, 'delta degrees'])
                for i in range(int(np.sqrt(self.sampled_features.shape[1]))):
                    feature_type3_name.append(['Pose ', i + 1, 'vs prev. time', 'delta pixels'])
                multi_columns = np.vstack((feature_type1_name, feature_type2_name, feature_type3_name))
                features_df = pd.DataFrame(self.sampled_features, columns=multi_columns)
                assignments_data = np.concatenate([self.assignments.reshape(len(self.assignments), 1),
                                                   self.soft_assignments.reshape(len(self.soft_assignments), 1),
                                                   ], axis=1)
                multi_columns2 = pd.MultiIndex.from_tuples([('HDBSCAN', 'Assignment'),
                                                            ('HDBSCAN*SOFT', 'Assignment')],
                                                           names=['Type', 'Frame@10Hz'])
                assignments_df = pd.DataFrame(assignments_data, columns=multi_columns2)
                training_data = pd.concat((features_df, assignments_df), axis=1)
                training_data.index.name = 'Frame@10hz'
                training_data.to_csv(
                    (os.path.join(self.working_dir, str.join('', (self.prefix, '_mapping.csv')))),
                    index=True, chunksize=10000, encoding='utf-8')
            if any('Assignment probabilities' in o for o in self.options):
                multi_columns = [str.join('', ('Group', str(i), '_probability'))
                                 for i in range(len(np.unique(self.soft_assignments)))]
                assign_prob_df = pd.DataFrame(self.assign_prob, columns=multi_columns)
                assign_prob_df.index.name = 'Frame@10hz'
                assign_prob_df.to_csv(
                    (os.path.join(self.working_dir, str.join('', (self.prefix, '_assign_prob.csv')))),
                    index=True, chunksize=10000, encoding='utf-8')
            st.balloons()
