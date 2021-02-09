import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from utilities.load_data import appdata
from utilities.processing import data_processing
import sys, getopt
from ast import literal_eval


def limb_trajectory(path, name, animal_idx, bp, t_range):
    appdata_ = appdata(path, name)
    _, _, _, data_new, fs_labels = appdata_.load_predictions()
    limbs = []
    labels = fs_labels[animal_idx][t_range[0]:t_range[1]]
    for b in range(len(bp)):
        limb = []
        for t in range(t_range[0], t_range[1]):
            limb.append(np.linalg.norm(data_new[animal_idx][t, bp[b]*2:bp[b]*2+2] -
                                       data_new[animal_idx][t - 1, bp[b]*2:bp[b]*2+2]))
        limbs.append(np.array(limb))
    return labels, limbs


def plot_trajectory(limbs, labels, t_range, ord1, ord2, c, fig_size, fig_format, outpath):
    proc_limb = []
    for l in range(len(limbs)):
        proc_data = data_processing(limbs[l])
        proc_limb.append(proc_data.boxcar_center(5))
    figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    plt.subplot()
    plt.subplot(211)
    ax1 = plt.subplot(2, 1, 1)
    for o in range(len(ord1)):
        if o > 0:
            a =  0.3
        else:
            a = 1
        ax1.plot(proc_limb[ord1[o]], linewidth=8, color=c[0], alpha=a)
    ax1 = plt.gca()
    # ax.set_xlim(43 - 30, 43 + 30)
    # ax.set_ylim(0, max(
    #     np.concatenate((boxcar_center(Rforelimb, 5)[43 - 30:43 + 30], boxcar_center(Lforelimb, 5)[43 - 30:43 + 30],
    #                     boxcar_center(Rhindlimb, 5)[43 - 30:43 + 30], boxcar_center(Lhindlimb, 5)[43 - 30:43 + 30]))))
    ax1.set_axis_off()

    ax2 = plt.subplot(2, 1, 2)
    for o in range(len(ord2)):
        if o > 0:
            a = 0.3
        else:
            a = 1
        ax2.plot(proc_limb[ord2[o]], linewidth=8, color=c[1], alpha=a)
    ax2 = plt.gca()

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['bottom'].set_visible(True)
    ax2.spines['bottom'].set_linewidth(4)
    ax1.spines['left'].set_visible(True)
    ax2.spines['left'].set_visible(True)
    plt.gca().invert_yaxis()

    # plt.axvline(x=6 * 8, linewidth=4, color='k')
    # plt.axvline(x=43, linewidth=4, color='k')
    ax1.tick_params(length=24, width=4)
    ax2.tick_params(length=24, width=4)
    ax2.xaxis.set_ticklabels([])
    # ax.set_xticks(range(43 - 30, 43 + 30, 15))
    plt.savefig(str.join('', (outpath, 'start{}_end{}_limb_trajectory.'.format(*t_range), fig_format)),
                format=fig_format, transparent=True)


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
    labels, limbs = limb_trajectory(path, name, int(animal_idx), literal_eval(bp), literal_eval(t_range))
    plot_trajectory(limbs, labels, literal_eval(t_range), literal_eval(order1), literal_eval(order2), literal_eval(c),
                    (8.5, 16), fig_format, outpath)


if __name__ == '__main__':
    main(sys.argv)



