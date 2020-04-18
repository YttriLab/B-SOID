"""
Visualization functions and saving plots.
"""
import os
from bsoid_py.config import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn
import pandas as pd
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import time
matplotlib_axes_logger.setLevel('ERROR')


def plot_tsne3d(data):
    """ Plot trained_tsne
    :param data: trained_tsne
    """
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_x, tsne_y, tsne_z, s=1, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(70, 135)
    plt.title('Embedding of the training set by t-SNE')
    plt.show()


def plot_classes(data, assignments):
    """ Plot trained_tsne for EM-GMM assignments
    :param data: trained_tsne
    :param assignments: EM-GMM assignments
    """
    uk = list(np.unique(assignments))
    R = np.linspace(0, 1, len(uk))
    cmap = plt.cm.get_cmap("Spectral")(R)
    tsne_x, tsne_y, tsne_z = data[:, 0], data[:, 1], data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for g in np.unique(assignments):
        idx = np.where(np.array(assignments) == g)
        ax.scatter(tsne_x[idx], tsne_y[idx], tsne_z[idx], c=cmap[g],
                   label=g, s=0.5, marker='o', alpha=0.8)
    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.view_init(70, 135)
    plt.title('Assignments by GMM')
    plt.legend(ncol=3)
    plt.show()
    timestr = time.strftime("_%Y%m%d_%H%M")
    my_file = 'GMM_assign_'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))


def plot_accuracy(scores):
    """
    :param scores: 1D array, cross-validated accuracies for SVM classifier.
    """
    fig = plt.figure(facecolor='w', edgecolor='k')
    fig.suptitle("Performance on {} % data".format(HLDOUT * 100))
    ax = fig.add_subplot(111)
    ax.boxplot(scores, notch=None)
    x = np.random.normal(1, 0.04, size=len(scores))
    plt.scatter(x, scores, s=40, c = 'r', alpha=0.5)
    ax.set_xlabel('SVM RBF kernel')
    ax.set_ylabel('Accuracy')
    plt.show()
    timestr = time.strftime("_%Y%m%d_%H%M")
    my_file = 'SVM_accuracy_'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))


def plot_cm(cm):
    """
    :param conf_mat: 2D array, cross-validated SVM accuracies for each class normalized.
    """
    timestr = time.strftime("_%Y%m%d_%H%M")
    fig = plt.figure()
    fig.suptitle("Confusion matrix (raw) on {} % data".format(HLDOUT * 100))
    df_cm = pd.DataFrame(cm)
    sn.heatmap(df_cm, annot=True)
    plt.show()
    my_file = 'SVM_CM_counts_'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    cm_norm = cm / cm.astype(np.float).sum(axis=1)
    fig = plt.figure()
    fig.suptitle("Confusion matrix (normalized) on {} % data".format(HLDOUT * 100))
    df_cm_norm = pd.DataFrame(cm_norm)
    sn.heatmap(df_cm_norm, annot=True)
    plt.show()
    my_file2 = 'SVM_CM_norm_'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file2, timestr, '.svg'))))
    return


def plot_durhist(lengths, grp):
    """
    :param lengths: 1D array, run lengths of each bout.
    :param grp: 1D array, corresponding label.
    """
    timestr = time.strftime("_%Y%m%d_%H%M")
    fig, ax = plt.subplots()
    R = np.linspace(0, 1, len(np.unique(grp)))
    cmap = plt.cm.get_cmap("Spectral")(R)
    for i in range(0, len(np.unique(grp))):
        fig.suptitle("Duration histogram of {} behaviors".format(len(np.unique(TM))))
        x = lengths[np.where(grp == i)]
        ax.hist(x, density=True, color=cmap[i], alpha=0.3, label='Group {}'.format(i))
    plt.legend(loc='upper right')
    plt.show()
    my_file = 'DurationHistogram_100msbins_'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    return


def plot_tmat(TM):
    """
    :param TM: 2D array, Transition matrix of behaviors.
    """
    timestr = time.strftime("_%Y%m%d_%H%M")
    fig = plt.figure()
    fig.suptitle("Transition matrix of {} behaviors".format(len(np.unique(TM))))
    df_tm = pd.DataFrame(TM)
    sn.heatmap(df_tm, annot=True)
    plt.xlabel("Next 100ms behavior")
    plt.ylabel("Current 100ms behavior")
    plt.show()
    my_file = 'TransitionMatrix_100msbins_'
    fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
    return


def plot_feats(feats, labels):
    """
    :param feats: list, features for multiple sessions
    :param labels: list, labels for multiple sessions
    """
    result = isinstance(labels, list)
    timestr = time.strftime("_%Y%m%d_%H%M")
    if result:
        for k in range(0, len(feats)):
            labels_k = np.array(labels[k])
            feats_k = np.array(feats[k])
            R = np.linspace(0, 1, len(np.unique(labels_k)))
            color = plt.cm.get_cmap("Spectral")(R)
            feat_ls = ("Distance between snout & center forepaw", "Distance between snout & center hind paw",
                       "forepaw distance", "Body length", "Angle", "Snout speed", "Proximal tail speed")
            for j in range(0, feats_k.shape[0]):
                fig = plt.figure(facecolor='w', edgecolor='k')
                for i in range(0, len(np.unique(labels_k))):
                    plt.subplot(len(np.unique(labels_k)), 1, i + 1)
                    if j == 2 or j == 3 or j == 5 or j == 6:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(0, np.mean(feats_k[j, :]) + 2 * np.std(feats_k[j, :]), num=50),
                                 range=(0, np.mean(feats_k[j, :]) + 2 * np.std(feats_k[j, :])),
                                 color=color[i], density=True)
                        fig.suptitle("{} pixels".format(feat_ls[j]))
                        plt.xlim(0, np.mean(feats_k[j, :]) + 2 * np.std(feats_k[j, :]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                    else:
                        plt.hist(feats_k[j, labels_k == i],
                                 bins=np.linspace(np.mean(feats_k[j, :]) - 2 * np.std(feats_k[j, :]),
                                                  np.mean(feats_k[j, :]) + 2 * np.std(feats_k[j, :]), num=50),
                                 range=(np.mean(feats_k[j, :]) - 2 * np.std(feats_k[j, :]),
                                        np.mean(feats_k[j, :]) + 2 * np.std(feats_k[j, :])),
                                 color=color[i], density=True)
                        plt.xlim(np.mean(feats_k[j, :]) - 2 * np.std(feats_k[j, :]),
                                 np.mean(feats_k[j, :]) + 2 * np.std(feats_k[j, :]))
                        fig.suptitle("{} pixels".format(feat_ls[j]))
                        if i < len(np.unique(labels_k)) - 1:
                            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                my_file = 'Session{}_feature{}_histogram_'.format(k + 1, j + 1)
                fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
                fig.savefig(os.path.join(OUTPUT_PATH, my_file))
            plt.show()
    else:
        R = np.linspace(0, 1, len(np.unique(labels)))
        color = plt.cm.get_cmap("Spectral")(R)
        feat_ls = ("Distance between snout & center forepaw", "Distance between snout & center hind paw",
                       "forepaw distance", "Body length", "Angle", "Snout speed", "Proximal tail speed")
        for j in range(0, feats.shape[0]):
            fig = plt.figure(facecolor='w', edgecolor='k')
            for i in range(0, len(np.unique(labels))):
                plt.subplot(len(np.unique(labels)), 1, i + 1)
                if j == 2 or j == 3 or j == 5 or j == 6:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(0, np.mean(feats[j, :]) + 2 * np.std(feats[j, :]), num=50),
                             range=(0, np.mean(feats[j, :]) + 2 * np.std(feats[j, :])),
                             color=color[i], density=True)
                    fig.suptitle("{} pixels".format(feat_ls[j]))
                    plt.xlim(0, np.mean(feats[j, :]) + 2 * np.std(feats[j, :]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                else:
                    plt.hist(feats[j, labels == i],
                             bins=np.linspace(np.mean(feats[j, :]) - 2 * np.std(feats[j, :]),
                                              np.mean(feats[j, :]) + 2 * np.std(feats[j, :]), num=50),
                             range=(np.mean(feats[j, :]) - 2 * np.std(feats[j, :]),
                                    np.mean(feats[j, :]) + 2 * np.std(feats[j, :])),
                             color=color[i], density=True)
                    plt.xlim(np.mean(feats[j, :]) - 2 * np.std(feats[j, :]),
                             np.mean(feats[j, :]) + 2 * np.std(feats[j, :]))
                    fig.suptitle("{} pixels".format(feat_ls[j]))
                    if i < len(np.unique(labels)) - 1:
                        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            my_file = 'feature{}_histogram_'.format(j + 1)
            fig.savefig(os.path.join(OUTPUT_PATH, str.join('', (my_file, timestr, '.svg'))))
        plt.show()



def main():
    return


if __name__ == '__main__':
    main()
