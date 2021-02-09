import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utilities.load_data import load_sav
import sys, getopt


def plot_boxplot(algo, data, c, fig_size, fig_format, outpath):
    figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    ax = plt.subplot()
    sns.boxplot(data=np.array(data), orient='h', width=0.7, medianprops={'color': 'white'}, color=c, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(3)
    ax.tick_params(length=24, width=3)
    ax.set_xlim(0.7, 1)
    ax.set_xticks(np.arange(0.70, 1.01, 0.1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.savefig(str.join('', (outpath, algo, '_frameshift_coherence.', fig_format)), format=fig_format, transparent=True)


def main(argv):
    path = None
    name = None
    vname = None
    algorithm = None
    c = None
    fig_format = None
    outpath = None
    options, args = getopt.getopt(
        argv[1:],
        'p:f:v:a:c:m:o:',
        ['path=', 'file=', 'variable=', 'algorithm=', 'color=', 'format=', 'outpath='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-f', '--file'):
            name = option_value
        elif option_key in ('-v', '--variable'):
            vname = option_value
        elif option_key in ('-a', '--algorithm'):
            algorithm = option_value
        elif option_key in ('-c', '--color'):
            c = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('VARIABLE   :', vname)
    print('ALGORITHM   :', algorithm)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    coherence_reordered = load_sav(path, name, vname)
    plot_boxplot(algorithm, np.array(coherence_reordered).T, c, (6, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)

