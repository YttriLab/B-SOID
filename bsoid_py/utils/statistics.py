"""
Summary statistics
"""

from bsoid_py.config import *
from bsoid_py.utils.visuals import plot_tmat
import numpy as np
import time
import pandas as pd
import os
import matplotlib.pyplot as plt


def transition_matrix(labels):
    n = 1 + max(labels)
    TM = [[0] * n for _ in range(n)]
    for (i, j) in zip(labels, labels[1:]):
        TM[i][j] += 1
    for row in TM:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]
    df_tm = pd.DataFrame(TM)
    return df_tm


def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]


def behv_time(labels):
    beh_t = []
    for i in range(0, len(np.unique(labels))):
        t = np.sum(labels == i) / labels.shape[0]
        beh_t.append(t)
    return beh_t


def behv_dur(labels, output_path=OUTPUT_PATH):
    lengths, pos, grp = rle(labels)
    df_lengths = pd.DataFrame(lengths, columns={'Run lengths'})
    df_grp = pd.DataFrame(grp, columns={'B-SOiD labels'})
    df_pos = pd.DataFrame(pos, columns={'Start time (10Hz frames)'})
    runlengths = [df_grp, df_pos, df_lengths]
    runlen_df = pd.concat(runlengths, axis=1)
    timestr = time.strftime("_%Y%m%d_%H%M")
    runlen_df.to_csv((os.path.join(output_path, str.join('', ('bsoid_runlengths', timestr, '.csv')))),
                     index=True, chunksize=10000, encoding='utf-8')
    beh_t = behv_time(labels)
    dur_means = []
    dur_quant0 = []
    dur_quant1 = []
    dur_quant2 = []
    dur_quant3 = []
    dur_quant4 = []
    for i in range(0, len(np.unique(grp))):
        dur_means.append(np.mean(lengths[np.where(grp == i)]))
        dur_quant0.append(np.quantile(lengths[np.where(grp == i)], 0.1))
        dur_quant1.append(np.quantile(lengths[np.where(grp == i)], 0.25))
        dur_quant2.append(np.quantile(lengths[np.where(grp == i)], 0.5))
        dur_quant3.append(np.quantile(lengths[np.where(grp == i)], 0.75))
        dur_quant4.append(np.quantile(lengths[np.where(grp == i)], 0.9))
    alldata = np.concatenate([np.array(beh_t).reshape(len(np.array(beh_t)), 1),
                              np.array(dur_means).reshape(len(np.array(dur_means)), 1),
                              np.array(dur_quant0).reshape(len(np.array(dur_quant0)), 1),
                              np.array(dur_quant1).reshape(len(np.array(dur_quant1)), 1),
                              np.array(dur_quant2).reshape(len(np.array(dur_quant2)), 1),
                              np.array(dur_quant3).reshape(len(np.array(dur_quant3)), 1),
                              np.array(dur_quant4).reshape(len(np.array(dur_quant4)), 1)], axis=1)
    micolumns = pd.MultiIndex.from_tuples([('Stats', 'Percent of time'),
                                           ('', 'Mean duration (100ms)'), ('', '10th Percentile.'),
                                           ('', '25th Percentile.'), ('', '50th Percentile.'),
                                           ('', '75th Percentile.'), ('', '90th Percentile.')],
                                          names=['', 'B-SOiD labels'])
    dur_stats = pd.DataFrame(alldata, columns=micolumns)
    return dur_stats


def main(labels, output_path=OUTPUT_PATH):
    dur_stats = behv_dur(labels, output_path)
    TM = transition_matrix(labels)
    plot_tmat(TM)
    return dur_stats, TM
