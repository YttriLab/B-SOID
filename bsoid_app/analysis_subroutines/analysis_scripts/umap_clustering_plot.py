import getopt
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.pyplot import figure

from analysis_subroutines.analysis_utilities.load_data import appdata

matplotlib_axes_logger.setLevel('ERROR')


def plot_enhanced_umap(path, name, fig_size, fig_format='png', outpath=os.getcwd(), save=True):
    appdata_ = appdata(path, name)
    f_10fps_sub, train_embeddings = appdata_.load_embeddings()
    min_cluster_range, assignments, soft_clusters, soft_assignments = appdata_.load_clusters()
    if save:
        m_size = 50
        lg_size = 8
        lg_text_sz = 12
    else:
        m_size = 0.2
        lg_size = 4
        lg_text_sz = 8
    uk = list(np.unique(assignments))
    r = np.linspace(0, 1, len(uk) - 1)
    cmap = plt.cm.get_cmap("Spectral")(r)
    umap_x, umap_y = train_embeddings[:, 0], train_embeddings[:, 1]
    fig = figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    for g in np.unique(assignments):
        if g >= 0:
            idx = np.where(np.array(assignments) == g)
            ax.scatter(umap_x[idx], umap_y[idx], c=cmap[g],
                       label=g, s=m_size, marker='o', alpha=0.6)
    plt.legend(ncol=4, loc=0, prop={'family': 'Helvetica', 'size': lg_text_sz}, markerscale=lg_size)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if save:
        ax.spines['top'].set_linewidth(3)
        ax.spines['right'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        ax.tick_params(length=9, width=3)
        plt.savefig(str.join('', (outpath, '{}'.format(name), '_umap_enahnced_clustering.', fig_format)),
                    format=fig_format, transparent=True)
    else:
        return fig, ax


def main(argv):
    path = None
    name = None
    fig_format = None
    outpath = None
    options, args = getopt.getopt(
        argv[1:],
        'p:f:m:o:',
        ['path=', 'file=', 'format=', 'outpath='])
    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-f', '--file'):
            name = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    plot_enhanced_umap(path, name, (16, 11), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
