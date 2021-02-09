import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utilities.load_data import load_mat
import sys, getopt
from ast import literal_eval
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns

def add_subplot_axes(ax, rect, axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height], facecolor=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



def plot_neural_fs_adv(var, data1, data2, order, c, x_range, bn, tk, fig_size, fig_format, outpath):
    fig = figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    axes = []
    Values1 = []
    Values2  =[]
    count1 = []
    count2 = []
    subpos = [0.5, 0.6, 0.3, 0.3]
    for i in range(len(data1)):
        axes.append(fig.add_subplot(3, 4, order[i]+1))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        values1, base = np.histogram(np.concatenate(data1[i].T), bins=np.arange(0, x_range[1]+0.01, bn),
                                     weights=np.ones(len(data1[i].T)) / len(data1[i].T), density=False)
        values2, base = np.histogram(np.concatenate(data2[i].T), bins=np.arange(0, x_range[1]+0.01, bn),
                                     weights=np.ones(len(data2[i].T)) / len(data2[i].T), density=False)
        Values1.append(np.append(values1, 0))
        Values2.append(np.append(values2, 0))
        count1.append(data1[i].shape[1])
        count2.append(data2[i].shape[1])
    for i, axis in enumerate(axes):
        axis.set_xlim(x_range[0], x_range[1])
        # axis.plot(base, Values1[i], color=c[0], marker='None', linestyle='-', linewidth=2)
        # axis.plot(base, Values2[i], color=c[1], marker='None', linestyle='-', linewidth=2)
        plt.hist(np.concatenate(data1[i].T), bins=np.arange(0, x_range[1]+0.01, bn),
                 weights=np.ones(len(data1[i].T)) / len(data1[i].T), density=False)
        plt.hist(np.concatenate(data2[i].T), bins=np.arange(0, x_range[1]+0.01, bn),
                 weights=np.ones(len(data2[i].T)) / len(data2[i].T), density=False)
        subax1 = add_subplot_axes(axis, subpos)
        subax1.bar(np.arange(0, 2), [count1[i], count2[i]], color=c)
        axis.set_xticks(np.arange(x_range[0], x_range[1]+0.1, (x_range[1]-x_range[0])/tk))
        axis.set_yticks(np.arange(0, 0.55, 0.5))
        axis.xaxis.set_ticklabels([])
        axis.yaxis.set_ticklabels([])
        subax1.xaxis.set_ticklabels([])
        subax1.set_xticks([])
        subax1.yaxis.set_ticklabels([])
        subax1.set_yticks([])
        axis.spines['top'].set_visible(False)
        axis.spines['top'].set_linewidth(3)
        axis.spines['right'].set_visible(False)
        axis.spines['right'].set_linewidth(3)
        axis.spines['bottom'].set_visible(True)
        axis.spines['bottom'].set_linewidth(3)
        axis.spines['left'].set_visible(True)
        axis.spines['left'].set_linewidth(3)
        axis.spines['top'].set_color('k')
        axis.spines['right'].set_color('k')
        axis.spines['bottom'].set_color('k')
        axis.spines['left'].set_color('k')
        axis.tick_params(length=10, width=3)
    plt.savefig(str.join('', (outpath, '{}_fsvsnonfs_duration_counts.'.format(var[0]), fig_format)),
                format=fig_format, transparent=True)


def main(argv):
    path = None
    var = None
    c = None
    x_range = None
    order = None
    fig_format = None
    outpath = None
    options, args = getopt.getopt(
        argv[1:],
        'p:v:c:r:O:m:o:',
        ['path=', 'variables=', 'colors=', 'range=', 'order=', 'format=', 'outpath='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-v', '--variables'):
            var = option_value
        elif option_key in ('-c', '--colors'):
            c = option_value
        elif option_key in ('-r', '--range'):
            x_range = option_value
        elif option_key in ('-O', '--order'):
            order = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('VARIABLES   :', var)
    print('COLOR   :', c)
    print('RANGE   :', x_range)
    print('ORDER   :', order)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    mat = load_mat(path)
    data1 = mat['L5nb_nonfsdurs'][0]
    data2 = mat['L5nb_fsdurs'][0]
    plot_neural_fs_adv(literal_eval(var), data1, data2, literal_eval(order), literal_eval(c), literal_eval(x_range),
             0.01, 3, (16, 12), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
