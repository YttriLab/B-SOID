import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utilities.load_data import load_sav
import sys, getopt
from ast import literal_eval


def plot_cdf(var, vname, data, c, x_range, bnct, tk, leg, fig_size, fig_format, outpath):
    figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    ax = plt.axes()
    values1, base = np.histogram(data[0], bins=np.arange(x_range[0], x_range[1]+0.5, 0.05),
                                                           # np.percentile(data[0], 95),
                                                           # num=bnct),
                                 weights=np.ones(len(data[0])) / len(data[0]), density=False)
    values2, base = np.histogram(data[1], bins=np.arange(x_range[0], x_range[1]+0.5, 0.05),
                                                           # np.percentile(data[0], 95),
                                                           # num=bnct),
                                 weights=np.ones(len(data[1])) / len(data[1]), density=False)
    values1 = np.append(values1, 0)
    values2 = np.append(values2, 0)

    # figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    # ax = plt.axes()
    # values1, base = np.histogram(data1[0], bins=np.linspace(x_range[0], data1[0].max(),
    #                                                        # np.percentile(data[0], 95),
    #                                                        num=bnct),
    #                              weights=np.ones(len(data1[0])) / len(data1[0]), density=False)
    # values2, base = np.histogram(data1[1], bins=np.linspace(x_range[0], data1[1].max(),
    #                                                        # np.percentile(data[0], 95),
    #                                                        num=bnct),
    #                              weights=np.ones(len(data1[1])) / len(data1[1]), density=False)
    # values1 = np.append(values1, 0)
    # values2 = np.append(values2, 0)

    ax.plot(base, np.cumsum(values1) / np.cumsum(values1)[-1],
            color=c[0], marker='None', linestyle='-',
            label="A2A Ctrl.", linewidth=8)
    ax.plot(base, np.cumsum(values2) / np.cumsum(values2)[-1],
            color=c[1], marker='None', linestyle='-',
            label="A2A Casp.", linewidth=8)

    # ax.set_xlim(np.percentile(data[0], 5), np.percentile(data[0], 95))
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(linestyle='-', linewidth=5, axis='both')
    ax.set_xticks(np.arange(x_range[0], x_range[1]+0.1, (x_range[1]-x_range[0])/tk))
    if leg:
        lgnd = plt.legend(loc=0, prop={'family': 'Helvetica', 'size': 60})
        lgnd.legendHandles[0]._legmarker.set_markersize(8)
        lgnd.legendHandles[1]._legmarker.set_markersize(8)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(5)
    ax.spines['top'].set_color('k')
    ax.spines['right'].set_color('k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.tick_params(length=25, width=5)
    plt.savefig(str.join('', (outpath, '{}_{}_cdf.'.format(var, vname), fig_format)),
                format=fig_format, transparent=True)


def main(argv):
    path = None
    name = None
    var = None
    vname = None
    bp = None
    c = None
    x_range = None
    leg = None
    fig_format = None
    outpath = None
    options, args = getopt.getopt(
        argv[1:],
        'p:n:v:V:b:c:r:l:m:o:',
        ['path=', 'file=', 'variables=', 'variable_name', 'bodypart=',
         'colors=', 'range=', 'legend=', 'format=', 'outpath='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-n', '--file'):
            name = option_value
        elif option_key in ('-v', '--variables'):
            var = option_value
        elif option_key in ('-V', '--variable_name'):
            vname = option_value
        elif option_key in ('-b', '--bodypart'):
            bp = option_value
        elif option_key in ('-c', '--colors'):
            c = option_value
        elif option_key in ('-r', '--range'):
            x_range = option_value
        elif option_key in ('-l', '--legend'):
            leg = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('VARIABLES   :', var)
    print('VARIABLE NAME   :', vname)
    print('BODYPART   :', bp)
    print('COLOR   :', c)
    print('RANGE   :', x_range)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    kin_data = load_sav(path, name, var)
    if vname == 'Distance':
        conv = 0
    elif vname == 'Speed':
        conv = 1
    elif vname == 'Duration':
        conv = 2
    if conv == 0:
        data = [np.concatenate(kin_data[0][int(bp)] / 23.5126),
                np.concatenate(kin_data[1][int(bp)] / 23.5126)]
        # data1 = [data[0][data[0] > 0.5], data[1][data[1] > 0.5]]
    elif conv == 1:
        data = [np.concatenate(kin_data[2][int(bp)] * 60 / 23.5126),
                np.concatenate(kin_data[3][int(bp)] * 60 / 23.5126)]
        # data1 = [data[0][data[0] > 3], data[1][data[1] > 3]]
    elif conv == 2:
        data = [kin_data[4][int(bp)] / 60,
                kin_data[5][int(bp)] / 60]
        # data1 = data
    plot_cdf(var, vname, data, literal_eval(c), literal_eval(x_range),
             50, 4, int(leg), (16, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
