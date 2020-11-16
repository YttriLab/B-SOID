import getopt
import sys

import numpy as np

from analysis_utilities.load_data import load_sav
from analysis_utilities.visuals import plot_coherence_boxplot


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
    plot_coherence_boxplot(algorithm, np.array(coherence_reordered).T, c, (6, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)

