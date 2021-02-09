import seaborn as sns
import matplotlib.colors as mc
import colorsys
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utilities.load_data import load_sav
import sys, getopt
from ast import literal_eval

def lighten_color(color, amount=0):
    # --------------------- SOURCE: @IanHincks ---------------------
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_boxplot(algo, data, c, fig_size, fig_format, outpath):
    figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    ax = plt.subplot()
    sns.set_palette(sns.color_palette(c))
    sns.boxplot(data=np.array(data), orient='h', width=0.7, ax=ax)
    for i, artist in enumerate(ax.artists):
        col = lighten_color(artist.get_facecolor(), 1.4)
        artist.set_edgecolor('k')
        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)
            line.set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_linewidth(3)
    ax.tick_params(length=24, width=3)
    ax.set_xlim(0.8, 1)
    ax.set_xticks(np.arange(0.80, 1.01, 0.1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.savefig(str.join('', (outpath, algo, '_Kfold_accuracy.', fig_format)), format=fig_format, transparent=True)


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
        ['path=', 'file=', 'variable=', 'algorithm=', 'colors=', 'format=', 'outpath='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-f', '--file'):
            name = option_value
        elif option_key in ('-v', '--variable'):
            vname = option_value
        elif option_key in ('-a', '--algorithm'):
            algorithm = option_value
        elif option_key in ('-c', '--colors'):
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
    print('COLORS   :', c)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    _, accuracy_ordered = load_sav(path, name, vname)
    plot_boxplot(algorithm, accuracy_ordered, literal_eval(c), (6, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
