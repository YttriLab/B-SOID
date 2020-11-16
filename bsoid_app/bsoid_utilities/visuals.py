import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from sklearn.metrics import plot_confusion_matrix

matplotlib_axes_logger.setLevel('ERROR')


def plot_bar(sub_threshold):
    st.write('If the below __% noise__ (y-axis) is unreasonable, consider refining pose-estimation software.')
    sub_threshold_df = pd.DataFrame(sub_threshold)
    col1, col2 = st.beta_columns([3, 2])
    col1.line_chart(sub_threshold_df)
    col2.write(sub_threshold_df)


def show_data_table(raw_input_data, processed_input_data):
    try:
        ID = int(
            st.number_input('Enter data file _index__:', min_value=1, max_value=len(raw_input_data), value=1))
        st.write(raw_input_data[ID - 1])
        st.write(processed_input_data[ID - 1])
    except IndexError:
        pass


def plot_classes(data, assignments):
    """ Plot umap_embeddings for HDBSCAN assignments
    :param data: 2D array, umap_embeddings
    :param assignments: 1D array, HDBSCAN assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    cmap = plt.cm.get_cmap("Spectral")(R)
    umap_x, umap_y, umap_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(umap_x[idx], umap_y[idx], umap_z[idx], c=cmap[g],
                   label=g, s=0.4, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    plt.legend(ncol=3, markerscale=6)
    return fig, plt


def plot_accuracy(scores):
    """
    :param scores: 1D array, cross-validated accuracies for MLP classifier.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle("Performance on {} % data".format(0.2 * 100))
    ax = sns.violinplot(data=scores, palette="muted", scale="count", inner="quartile", width=0.4, linewidth=2,
                       scale_hue=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(1)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_xlabel('RF classifier')
    ax.set_ylabel('Accuracy')
    return fig, ax


def plot_confusion(validate_clf, x_test, y_test):
    titles_options = [("Non-normalized confusion matrix", None), ("Normalized confusion matrix", 'true')]
    st.write(
        'Two confusion matrices - top: counts, bottom: probability with **true positives in diagonal**')
    confusion = []
    for title, normalize in titles_options:
        cm = plot_confusion_matrix(validate_clf, x_test, y_test, cmap=sns.cm.rocket_r, normalize=normalize)
        cm.ax_.set_title(title)
        confusion.append(cm.figure_)
    return confusion


def plot_tmat(tm: object):
    """
    :param tm: object, transition matrix data frame
    :param fps: scalar, camera frame-rate
    """
    fig = plt.figure()
    fig.suptitle("Transition matrix of {} behaviors".format(tm.shape[0]))
    sns.heatmap(tm, annot=True)
    plt.xlabel("Next frame behavior")
    plt.ylabel("Current frame behavior")
    return fig