import getopt
import sys
from ast import literal_eval

import numpy as np

from analysis_subroutines.analysis_utilities.load_data import load_sav
from analysis_utilities.visuals import plot_kinematics_cdf


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
    elif conv == 1:
        data = [np.concatenate(kin_data[2][int(bp)] * 60 / 23.5126),
                np.concatenate(kin_data[3][int(bp)] * 60 / 23.5126)]
    elif conv == 2:
        data = [kin_data[4][int(bp)] / 60,
                kin_data[5][int(bp)] / 60]
    plot_kinematics_cdf(None, var, vname, data, literal_eval(c), 50, 4, int(leg), (16, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
