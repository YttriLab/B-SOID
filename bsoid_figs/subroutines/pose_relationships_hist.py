import numpy as np
import itertools
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from utilities.load_data import appdata
import sys, getopt
from ast import literal_eval


def plot_pose_relationships(path, name, order, fig_size, fig_format, outpath):
    appdata_ = appdata(path, name)
    f_10fps_sub, _ = appdata_.load_embeddings()
    _, assignments, _, _ = appdata_.load_clusters()
    length_nm = []
    angle_nm = []
    disp_nm = []
    for i, j in itertools.combinations(range(0, int(np.sqrt(f_10fps_sub.shape[1]))), 2):
        length_nm.append(['distance between points:', i + 1, j + 1])
        angle_nm.append(['angular change for points:', i + 1, j + 1])
    for i in range(int(np.sqrt(f_10fps_sub.shape[1]))):
        disp_nm.append(['displacement for point:', i + 1, i + 1])
    keys = np.arange(len(length_nm) + len(angle_nm) + len(disp_nm))
    POSE_RELATIONSHIPS = OrderedDict({key: [] for key in keys})
    for m, feat_name in enumerate(length_nm):
        POSE_RELATIONSHIPS[m] = feat_name
    for n, feat_name in enumerate(angle_nm):
        POSE_RELATIONSHIPS[m + n + 1] = feat_name
    for o, feat_name in enumerate(disp_nm):
        POSE_RELATIONSHIPS[m + n + o + 2] = feat_name
    R = np.linspace(0, 1, len(np.unique(assignments)))
    cm = plt.cm.get_cmap("Spectral")(R)
    for f in range(f_10fps_sub.shape[1]):
        fig = figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
        fig.suptitle("{}".format(POSE_RELATIONSHIPS[f]), fontsize=30)
        k = 0
        for i in order:
            k += 1
            ax = plt.subplot(len(np.unique(assignments)), 1, k)
            if f <= m or f > m + n + 1:
                values, base = np.histogram(f_10fps_sub[assignments == i, f] / 23.5126,
                                            bins=np.linspace(0, np.mean(f_10fps_sub[assignments == i, f] / 23.5126) +
                                                             3 * np.std(f_10fps_sub[assignments == i, f] / 23.5126),
                                                             num=50),
                                            weights=np.ones(len(f_10fps_sub[assignments == i, f])) /
                                                    len(f_10fps_sub[assignments == i, f]),
                                            density=False)
                values = np.append(values, 0)
                ax.plot(base, values,
                        color=cm[k-1], marker='None', linestyle='-', linewidth=5)

                ax.set_xlim(0, np.mean(f_10fps_sub[:, f] / 23.5126) + 3 * np.std(f_10fps_sub[:, f] / 23.5126))
                if i < len(np.unique(assignments)) - 2:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labelsize=16)
                    ax.tick_params(axis='y', which='both', right=False, labelright=False, labelsize=16)
                else:
                    ax.tick_params(labelsize=16)
                    ax.set_xticks(np.linspace(0, np.mean(f_10fps_sub[:, f] / 23.5126) +
                                              3 * np.std(f_10fps_sub[:, f] / 23.5126), num=5))
                    fig.text(0.5, 0.07, 'Centimeters', ha='center', fontsize=16)
                    fig.text(0.03, 0.5, 'Probability', va='center', rotation='vertical', fontsize=16)
            else:
                values, base = np.histogram(f_10fps_sub[assignments == i, f],
                                            bins=np.linspace(np.mean(f_10fps_sub[assignments == i, f]) -
                                                             3 * np.std(f_10fps_sub[assignments == i, f]),
                                                             np.mean(f_10fps_sub[assignments == i, f]) +
                                                             3 * np.std(f_10fps_sub[assignments == i, f]), num=50),
                                            weights=np.ones(len(f_10fps_sub[assignments == i, f])) /
                                                    len(f_10fps_sub[assignments == i, f]),
                                            density=False)
                values = np.append(values, 0)
                ax.plot(base, values,
                        color=cm[k-1], marker='None', linestyle='-', linewidth=5)
                ax.set_xlim(np.mean(f_10fps_sub[:, f]) - 3 * np.std(f_10fps_sub[:, f]),
                            np.mean(f_10fps_sub[:, f]) + 3 * np.std(f_10fps_sub[:, f]))
                if i < len(np.unique(assignments)) - 2:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labelsize=16)
                    ax.tick_params(axis='y', which='both', right=False, labelright=False, labelsize=16)
                else:
                    ax.tick_params(labelsize=16)
                    ax.set_xticks(np.linspace(np.mean(f_10fps_sub[:, f]) -
                                              3 * np.std(f_10fps_sub[:, f]),
                                              np.mean(f_10fps_sub[:, f]) +
                                              3 * np.std(f_10fps_sub[:, f]), num=5))
                    fig.text(0.5, 0.07, 'Degrees', ha='center', fontsize=16)
                    fig.text(0.03, 0.5, 'Probability', va='center', rotation='vertical', fontsize=16)
        plt.savefig(str.join('', (outpath, '{}_{}_histogram.'.format(name, POSE_RELATIONSHIPS[f]), fig_format)),
                    format=fig_format, transparent=True)
        plt.close()
    return


def main(argv):
    path = None
    name = None
    order = None
    fig_format = None
    outpath = None
    options, args = getopt.getopt(
        argv[1:],
        'p:f:r:m:o:',
        ['path=', 'file=', 'order=', 'format=', 'outpath='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-f', '--file'):
            name = option_value
        elif option_key in ('-r', '--order'):
            order = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value

    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('ORDER   :', order)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    plot_pose_relationships(path, name, literal_eval(order), (11, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
