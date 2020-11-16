import getopt
import sys
from ast import literal_eval

import numpy as np

from analysis_subroutines.analysis_utilities.load_data import appdata
from analysis_subroutines.analysis_utilities.visuals import plot_trajectory


def limb_trajectory(path, name, animal_idx, bp, t_range):
    appdata_ = appdata(path, name)
    _, _, _, data_new, fs_labels = appdata_.load_predictions()
    _, _, _, soft_assignments = appdata_.load_clusters()
    limbs = []
    labels = fs_labels[animal_idx][t_range[0]:t_range[1]]
    for b in range(len(bp)):
        limb = []
        for t in range(t_range[0], t_range[1]):
            limb.append(np.linalg.norm(data_new[animal_idx][t, bp[b] * 2:bp[b] * 2 + 2] -
                                       data_new[animal_idx][t - 1, bp[b] * 2:bp[b] * 2 + 2]))
        limbs.append(np.array(limb, dtype=object))
    return labels, limbs, soft_assignments


def main(argv):
    path = None
    name = None
    animal_idx = None
    bp = None
    t_range = None
    order1 = None
    order2 = None
    c = None
    fig_format = None
    outpath = None
    options, args = getopt.getopt(
        argv[1:],
        'p:f:i:b:t:r:R:c:m:o:',
        ['path=', 'file=', 'animal_idx=', 'bodypart=', 'timerange=', 'order1=', 'order2=', 'colors=',
         'format=', 'outpath='])
    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-f', '--file'):
            name = option_value
        elif option_key in ('-i', '--animal_idx'):
            animal_idx = option_value
        elif option_key in ('-b', '--bodypart'):
            bp = option_value
        elif option_key in ('-t', '--timerange'):
            t_range = option_value
        elif option_key in ('-r', '--order1'):
            order1 = option_value
        elif option_key in ('-R', '--order1'):
            order2 = option_value
        elif option_key in ('-c', '--colors'):
            c = option_value
        elif option_key in ('-m', '--format'):
            fig_format = option_value
        elif option_key in ('-o', '--outpath'):
            outpath = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('ANIMAL INDEX  :', animal_idx)
    print('BODYPARTS   :', bp)
    print('TIME RANGE   :', t_range)
    print('TOP PLOT   :', order1)
    print('BOTTOM PLOT    :', order2)
    print('COLORS   :', c)
    print('FIG FORMAT   :', fig_format)
    print('OUT PATH   :', outpath)
    print('*' * 50)
    print('Plotting...')
    labels, limbs, soft_assignments = limb_trajectory(path, name, int(animal_idx), literal_eval(bp), literal_eval(t_range))
    plot_trajectory(limbs, labels, soft_assignments,
                    literal_eval(t_range), literal_eval(order1), literal_eval(order2), literal_eval(c),
                    (8.5, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)
