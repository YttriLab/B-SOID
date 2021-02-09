import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utilities.load_data import load_mat
import sys, getopt
from ast import literal_eval


def plot_cdf(data, c, x_range, fig_size, fig_format, outpath):
    figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    ax = plt.axes()
    values1, base = np.histogram(data[0], bins=np.arange(0, 3, 0.01),
                                 weights=np.ones(len(data[0])).reshape(len(data[0]), 1) / len(data[0]), density=False)
    values2, base = np.histogram(data[1], bins=np.arange(0, 3, 0.01),
                                 weights=np.ones(len(data[1])).reshape(len(data[1]), 1) / len(data[1]), density=False)
    values3, base = np.histogram(data[2], bins=np.arange(0, 3, 0.01),
                                 weights=np.ones(len(data[2])).reshape(len(data[2]), 1) / len(data[2]), density=False)
    values4, base = np.histogram(data[3], bins=np.arange(0, 3, 0.01),
                                 weights=np.ones(len(data[3])).reshape(len(data[3]), 1) / len(data[3]), density=False)
    values5, base = np.histogram(data[4], bins=np.arange(0, 3, 0.01),
                                 weights=np.ones(len(data[4])).reshape(len(data[4]), 1) / len(data[4]), density=False)
    values6, base = np.histogram(data[5], bins=np.arange(0, 3, 0.01),
                                 weights=np.ones(len(data[5])).reshape(len(data[5]), 1) / len(data[5]), density=False)
    values1 = np.append(values1, 0)
    values2 = np.append(values2, 0)
    values3 = np.append(values3, 0)
    values4 = np.append(values4, 0)
    values5 = np.append(values5, 0)
    values6 = np.append(values6, 0)
    ax.plot(base, np.cumsum(values5) / np.cumsum(values5)[-1],
            color=c[2], marker='None', linestyle='-',
            label="Shuff. same", linewidth=8)
    ax.plot(base, np.cumsum(values6) / np.cumsum(values6)[-1],
            color=c[2], marker='None', linestyle='--',
            label="Shuff. diff.", linewidth=8)
    ax.plot(base, np.cumsum(values1) / np.cumsum(values1)[-1],
            color=c[0], marker='None', linestyle='-',
            label="MM same", linewidth=6)
    ax.plot(base, np.cumsum(values2) / np.cumsum(values2)[-1],
            color=c[0], marker='None', linestyle='--',
            label="MM diff.", linewidth=6)
    ax.plot(base, np.cumsum(values3) / np.cumsum(values3)[-1],
            color=c[1], marker='None', linestyle='-',
            label="BSOiD same", linewidth=6)
    ax.plot(base, np.cumsum(values4) / np.cumsum(values4)[-1],
            color=c[1], marker='None', linestyle='--',
            label="BSOiD diff.", linewidth=6)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(0, 1)
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.set_xticks(np.arange(x_range[0], x_range[1]+0.1, (x_range[1]-x_range[0])/3))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    order = [2, 3, 4, 5, 0, 1]
    handles, labels = plt.gca().get_legend_handles_labels()
    lgnd = plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       loc=0, prop={'family': 'Helvetica', 'size': 48})
    lgnd.legendHandles[0]._legmarker.set_markersize(2)
    lgnd.legendHandles[1]._legmarker.set_markersize(2)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_visible(True)
    ax.spines['right'].set_linewidth(5)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(5)
    ax.tick_params(length=20, width=5)
    plt.savefig(str.join('', (outpath, 'mse_cdf.', fig_format)), format=fig_format, transparent=True)



def main(argv):
    path = None
    c = None
    x_range = None
    fig_format = None
    outpath = None
    options, args = getopt.getopt(
        argv[1:],
        'p:c:r:m:o:',
        ['path=', 'color=', 'range=', 'format=', 'outpath='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-c', '--color'):
            c = option_value
        elif option_key in ('-r', '--range'):
            x_range = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('COLOR   :', c)
    print('RANGE   :', x_range)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    mat = load_mat(path)
    data = [mat['mm_within_vec2'], mat['mm_between_vec2'],
            mat['bsf_within_vec2'], mat['bsf_between_vec2'],
            mat['sbsf_within_vec2'], mat['sbsf_between_vec2']]
    plot_cdf(data, literal_eval(c), literal_eval(x_range), (16, 13), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
