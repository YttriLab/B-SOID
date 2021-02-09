import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utilities.load_data import load_mat
import sys, getopt
from ast import literal_eval


def plot_heatmap(algo, data, c, c_range, discrete_n, delim, cl, fig_size, fig_format, outpath, cb=False):
    figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    ax = plt.subplot()
    cm = sns.diverging_palette(275, 35, center='light', s=100, l=70, n=5)
    # cm = sns.light_palette(c, n_colors=discrete_n)
    sns.heatmap(data, vmin=c_range[0], vmax=c_range[1], center=1, cmap=cm, cbar=cb, ax=ax)
    # sns.heatmap(data, vmin=c_range[0], vmax=c_range[1], cmap=cm, cbar=cb, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    for i in range(delim.shape[1]):
        plt.axvline(x=np.cumsum(delim)[i] - 1, linewidth=4, linestyle='--', color=cl)
        plt.axhline(y=np.cumsum(delim)[i], linewidth=4, linestyle='--', color=cl)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    if cb:
        cax = plt.gcf().axes[-1]
        cax.tick_params(length=20, width=5, color='k')
    plt.savefig(str.join('', (outpath, algo, '_mse_matrix.', fig_format)), format=fig_format, transparent=True)


def main(argv):
    path = None
    algorithm = None
    c = None
    c_range = None
    discrete_n = None
    cline = None
    fig_format = None
    outpath = None
    cb = None
    options, args = getopt.getopt(
        argv[1:],
        'p:a:c:r:n:l:m:o:b:',
        ['path=', 'algorithm=', 'color=', 'range=', 'discrete_n=', 'cline=', 'format=', 'outpath=', 'colorbar='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-a', '--algorithm'):
            algorithm = option_value
        elif option_key in ('-c', '--color'):
            c = option_value
        elif option_key in ('-r', '--range'):
            c_range = option_value
        elif option_key in ('-n', '--discrete_n'):
            discrete_n = option_value
        elif option_key in ('-l', '--cline'):
            cline = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value
        elif option_key in ('-b', '--colorbar'):
            cb = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('ALGORITHM   :', algorithm)
    print('COLOR   :', c)
    print('RANGE   :', c_range)
    print('DISCRETE COLORS   :', discrete_n)
    print('LINE COLOR   :', cline)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('COLORBAR   :', cb)
    print('*' * 50)
    print('Plotting...')
    mat = load_mat(path)
    if algorithm == 'MotionMapper':
        data = mat['mm_mat_norm2_']
        delim = mat['mm_mse_counts']
    elif algorithm == 'B-SOiD':
        data = mat['bsf_mat_norm2_']
        delim = mat['bsf_mse_counts']
    plot_heatmap(algorithm, data, c, literal_eval(c_range), int(discrete_n), delim, cline,
                 (16, 14), fig_format, outpath, bool(int(cb)))


if __name__ == '__main__':
    main(sys.argv)
