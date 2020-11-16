import colorsys
import glob
import os
import subprocess

import matplotlib as mpl
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from matplotlib.pyplot import figure

from analysis_subroutines.analysis_utilities.processing import data_processing

matplotlib_axes_logger.setLevel('ERROR')


def discrete_cmap(n, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, n))
    cmap_name = base.name + str(n)
    return base.from_list(cmap_name, color_list, n)


def lighten_color(color, amount=0):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_accuracy_boxplot(algo, data, c, fig_size, fig_format='png', outpath=os.getcwd(), save=True):
    fig = figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
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
    ax.spines['left'].set_visible(True)
    ax.set_yticks(range(0, len(c), 3))
    ax.set_xticks(np.arange(np.percentile(np.concatenate(data), 5),
                            np.percentile(np.concatenate(data), 95), 0.1))
    if save:
        ax.spines['left'].set_linewidth(3)
        ax.spines['bottom'].set_linewidth(3)
        ax.set_yticks(range(0, len(c), 3))
        ax.set_xticks(np.arange(np.percentile(np.concatenate(data), 5),
                                np.percentile(np.concatenate(data), 95), 0.1))
        ax.tick_params(length=9, width=3)
        ax.tick_params(length=9, width=3)
        ax.tick_params(labelsize=24)
        ax.tick_params(labelsize=24)
        plt.savefig(str.join('', (outpath, algo, '_Kfold_accuracy.', fig_format)), format=fig_format, transparent=False)
    else:
        return fig, ax


def plot_coherence_boxplot(algo, data, c, fig_size, fig_format, outpath):
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
    plt.savefig(str.join('', (outpath, algo, '_frameshift_coherence.', fig_format)),
                format=fig_format, transparent=True)


def plot_peaks(x, ax, ind):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(16, 8.5))
    ax.plot(x, 'k', lw=1)
    hfont = {'fontname': 'Helvetica'}
    if ind.size:
        label = 'peak'
        label = label + 's' if ind.size > 1 else label
        ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=4, ms=16,
                label='%d %s' % (ind.size, label))
        ax.legend(loc='best', framealpha=.5, numpoints=1, prop={'family': 'Helvetica', 'size': 24})
    ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel('Bout duration (frames)', fontsize=24, **hfont)
    ax.set_ylabel('Pose estimate $\Delta$ (pixels)', fontsize=24, **hfont)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.tick_params(labelsize=20)
    ticks_font = mpl.font_manager.FontProperties(family='Helvetica', size=20)
    for l in ax.get_xticklabels():
        l.set_fontproperties(ticks_font)


def plot_kinematics_cdf(ax, var, vname, data, c, bnct, tk, leg,
                        fig_size, fig_format='png', outpath=os.getcwd(), save=True):
    fig = figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
    if save:
        ax = plt.axes()
    values1, base = np.histogram(data[0], bins=np.linspace(np.percentile(np.concatenate(data), 1),
                                                           np.percentile(np.concatenate(data), 99), num=bnct),
                                 weights=np.ones(len(data[0])) / len(data[0]), density=False)
    values2, base = np.histogram(data[1], bins=np.linspace(np.percentile(np.concatenate(data), 1),
                                                           np.percentile(np.concatenate(data), 99), num=bnct),
                                 weights=np.ones(len(data[1])) / len(data[1]), density=False)
    values1 = np.append(values1, 0)
    values2 = np.append(values2, 0)
    if save:
        lwidth = 8
        lg_size = 8
    else:
        lwidth = 2
        lg_size = 1
    ax.plot(base, np.cumsum(values1) / np.cumsum(values1)[-1],
            color=c[0], marker='None', linestyle='-',
            label="Ctrl.", linewidth=lwidth)
    ax.plot(base, np.cumsum(values2) / np.cumsum(values2)[-1],
            color=c[1], marker='None', linestyle='-',
            label="Exp.", linewidth=lwidth)
    ax.set_xlim(np.percentile(np.concatenate(data), 2), np.percentile(np.concatenate(data), 98))
    ax.set_ylim(0, 1)
    if leg:
        lgnd = ax.legend(loc=4, prop={'family': 'Helvetica', 'size': 12})
        lgnd.legendHandles[0]._legmarker.set_markersize(lg_size)
        lgnd.legendHandles[1]._legmarker.set_markersize(lg_size)
    ax.set_xticks(np.arange(int(np.percentile(np.concatenate(data), 2)),
                            int(np.percentile(np.concatenate(data), 98)) + 0.1,
                            int((np.percentile(np.concatenate(data), 98) -
                                 np.percentile(np.concatenate(data), 2)) / tk)))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    if save:
        ax.spines['top'].set_linewidth(5)
        ax.spines['right'].set_linewidth(5)
        ax.spines['bottom'].set_linewidth(5)
        ax.spines['left'].set_linewidth(5)
        ax.tick_params(length=15, width=5)
        ax.tick_params(labelsize=24)
        plt.savefig(str.join('', (outpath, '/{}_{}_cdf.'.format(var, vname), fig_format)),
                    format=fig_format, transparent=True)
    else:
        return fig, ax


def plot_trajectory(limbs, labels, soft_assignments, t_range, ord1, ord2, c,
                        fig_size, fig_format='png', outpath=os.getcwd(), save=True):
        proc_limb = []
        for l in range(len(limbs)):
            proc_data = data_processing(limbs[l])
            proc_limb.append(proc_data.boxcar_center(5))
        transitions = 0
        transitions = np.vstack((transitions, (np.argwhere(np.diff(labels) != 0) + 1)))
        transitions = np.vstack((transitions, len(labels)))
        uk = list(np.unique(soft_assignments))
        r = np.linspace(0, 1, len(uk))
        cmap = plt.cm.get_cmap("Spectral")(r)
        fig = figure(num=None, figsize=fig_size, dpi=300, facecolor='w', edgecolor='k')
        plt.subplot()
        plt.subplot(211)
        ax1 = plt.subplot(2, 1, 1)
        if save:
            lwidth = 8
        else:
            lwidth = 3
        for o in range(len(ord1)):
            if o > 0:
                a = 0.3
            else:
                a = 1
            ax1.plot(proc_limb[ord1[o]], linewidth=lwidth, color=c[0], alpha=a)
            for t in range(len(transitions) - 1):
                for g in np.unique(soft_assignments):
                    if labels[transitions[t]] == g and o == 0:
                        ax1.axvspan(transitions[t], transitions[t + 1], color=cmap[g], alpha=0.2, lw=0)
                        plt.text(transitions[t], np.max([proc_limb[ord1[i]].max() for i in range(len(ord1))]),
                                 '{}'.format(g), fontsize=6)
        ax1 = plt.gca()
        ax1.get_xaxis().set_visible(False)
        ax2 = plt.subplot(2, 1, 2)
        for o in range(len(ord2)):
            if o > 0:
                a = 0.3
            else:
                a = 1
            ax2.plot(proc_limb[ord2[o]], linewidth=lwidth, color=c[1], alpha=a)
            for t in range(len(transitions) - 1):
                for g in np.unique(soft_assignments):
                    if labels[transitions[t]] == g:
                        ax2.axvspan(transitions[t], transitions[t + 1], color=cmap[g], alpha=0.2, lw=0)
        ax2 = plt.gca()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['bottom'].set_visible(True)
        ax1.spines['left'].set_visible(True)
        ax2.spines['left'].set_visible(True)
        plt.gca().invert_yaxis()
        ax2.set_xticks(range(0, len(labels), 40))
        ax1.set_yticks(range(0, int(np.percentile(np.concatenate(limbs), 98)), 5))
        ax2.set_yticks(range(0, int(np.percentile(np.concatenate(limbs), 98)), 5))
        if save:
            ax1.spines['left'].set_linewidth(4)
            ax2.spines['left'].set_linewidth(4)
            ax2.spines['bottom'].set_linewidth(4)
            ax2.set_xticks(range(0, len(labels), 40))
            ax1.set_yticks(range(0, int(np.percentile(np.concatenate(limbs), 98)), 5))
            ax2.set_yticks(range(0, int(np.percentile(np.concatenate(limbs), 98)), 5))
            ax1.tick_params(length=12, width=4)
            ax2.tick_params(length=12, width=4)
            ax1.tick_params(labelsize=24)
            ax2.tick_params(labelsize=24)
            plt.savefig(str.join('', (outpath, 'start{}_end{}_limb_trajectory.'.format(*t_range), fig_format)),
                        format=fig_format, transparent=False)
        else:
            return fig, ax1, ax2


def umap_scatter(embeds, assigns, mov_range, output_path, width, height):
    uk = list(np.unique(assigns))
    R = np.linspace(0, 1, len(uk))
    cmap = plt.cm.get_cmap("Spectral")(R)
    umap_x, umap_y = embeds[mov_range[0] - 150:mov_range[1], 0], embeds[mov_range[0] - 150:mov_range[1], 1]
    fig = figure(facecolor='k', edgecolor='w')
    fig.set_size_inches(width / 100, height / 100)
    ax = fig.add_subplot(111)
    ax.axes.axis([min(umap_x) - 0.2, max(umap_x) + 0.2, min(umap_y) - 0.2, max(umap_y) + 0.2])
    count = 0
    for i, j in enumerate(range(mov_range[0] - 150, mov_range[1])):
        if i < 150:
            alph = 0.5
            m_size = 30
        else:
            alph = 0.8
            m_size = 80
        for g in np.unique(assigns):
            if assigns[j] == g and assigns[j] >= 0:
                ax.scatter(umap_x[i], umap_y[i], c=cmap[g], edgecolors='w',
                           label=g, s=m_size, marker='o', alpha=alph)
        if i >= 150:
            for g in np.unique(assigns):
                if assigns[j] == g and assigns[j] >= 0:
                    txt = plt.text(1, 1, 'group {}'.format(g), c='white', horizontalalignment='center',
                                   verticalalignment='center', transform=ax.transAxes, fontsize=20,
                                   bbox=dict(facecolor=cmap[g], alpha=0.8))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.set_facecolor('black')
            ax.tick_params(length=6, width=2, color='white')
            count += 1
            plt.savefig(output_path + "/file%04d.png" % count)
            txt.set_visible(False)

    plt.close('all')
    subprocess.call([
        'ffmpeg', '-y', '-framerate', '10', '-i', output_path + '/file%04d.png',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        output_path + '/umap_enhanced_clustering{}_{}.mp4'.format(*mov_range)
    ])
    for file_name in glob.glob(output_path + "/*.png"):
        os.remove(file_name)


def trim_video(mov_path, mov_file, mov_range, mov_st_min, mov_st_sec, mov_sp_min, mov_sp_sec, output_path):
    print(mov_path, mov_file)
    subprocess.call([
        'ffmpeg', '-y', '-i', str.join('', (mov_path, '/', mov_file)),
        '-ss', '00:{}:{}'.format(mov_st_min, mov_st_sec), '-to', '00:{}:{}'.format(mov_sp_min, mov_sp_sec),
        output_path + '/video_trim2umap{}_{}.mp4'.format(*mov_range)
    ])
    subprocess.call([
        'ffmpeg', '-y', '-i', output_path + '/video_trim2umap{}_{}.mp4'.format(*mov_range),
        '-filter:v', 'fps=fps=10',
        output_path + '/video_trim2umap{}_{}_10fps.mp4'.format(*mov_range)
    ])


def video_umap(output_path, mov_range):
    subprocess.call([
        'ffmpeg', '-y', '-i', output_path + '/video_trim2umap{}_{}_10fps.mp4'.format(*mov_range),
        '-i', output_path + '/umap_enhanced_clustering{}_{}.mp4'.format(*mov_range),
        '-filter_complex', "[0]pad=iw+5:color=white[left];[left][1]hstack=inputs=2",
        output_path + '/sync_leftright_video2umap{}_{}.mp4'.format(*mov_range)
    ])



