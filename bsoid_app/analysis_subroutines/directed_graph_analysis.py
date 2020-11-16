import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st

from analysis_subroutines.analysis_utilities import statistics


class directed_graph:

    def __init__(self, working_dir, prefix, soft_assignments, folders, folder, new_predictions):
        st.subheader('(ALPHA) BEHAVIORAL DIRECTED GRAPH')
        self.working_dir = working_dir
        self.prefix = prefix
        self.soft_assignments = soft_assignments
        self.folders = folders
        self.folder = folder
        self.new_predictions = new_predictions
        self.node_sizes = []
        self.transition_matrix_norm = []

    def compute_dynamics(self):
        selected_flder = st.sidebar.selectbox('select folder', [*self.folders])
        try:
            indices = [i for i, s in enumerate(self.folder) if str(selected_flder) in s]
            tm_count_all = []
            tm_prob_all = []
            for idx in indices:
                runlen_df, dur_stats, tm_array, tm_df, tm_norm = statistics.main(self.new_predictions[idx],
                                                                                 len(np.unique(self.soft_assignments)))
                tm_count_all.append(tm_array)
                tm_prob_all.append(tm_norm)
            tm_count_mean = np.nanmean(tm_count_all, axis=0)
            tm_prob_mean = np.nanmean(tm_prob_all, axis=0)
            diag = [tm_count_mean[i][i] for i in range(len(tm_count_mean))]
            diag_p = np.array(diag) / np.array(diag).max()
            self.node_sizes = [50 * i for i in diag_p]
            transition_matrix = np.matrix(tm_prob_mean)
            np.fill_diagonal(transition_matrix, 0)
            self.transition_matrix_norm = transition_matrix / transition_matrix.sum(axis=1)
            nan_indices = np.isnan(self.transition_matrix_norm)
            self.transition_matrix_norm[nan_indices] = 0
        except:
            pass

    def plot(self):
        if st.checkbox('Show directed graph?', False, key='ds'):
            fig = plt.figure()
            graph = nx.from_numpy_matrix(self.transition_matrix_norm, create_using=nx.MultiDiGraph())
            node_position = nx.layout.spring_layout(graph, seed=0)
            edge_colors = [graph[u][v][0].get('weight') for u, v in graph.edges()]
            nodes = nx.draw_networkx_nodes(graph, node_position, node_size=self.node_sizes,
                                           node_color='blue')
            edges = nx.draw_networkx_edges(graph, node_position, node_size=self.node_sizes, arrowstyle='->',
                                           arrowsize=8, edge_color=edge_colors, edge_cmap=plt.cm.Blues, width=1.5)
            label_pos = [node_position[i] + 0.005 for i in range(len(node_position))]
            nx.draw_networkx_labels(graph, label_pos, font_size=10)
            pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
            pc.set_array(edge_colors)
            plt.colorbar(pc)
            ax = plt.gca()
            ax.set_axis_off()
            st.pyplot(fig)

    def main(self):
        self.compute_dynamics()
        self.plot()
