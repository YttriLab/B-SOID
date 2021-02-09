import numpy as np
import pandas as pd


def transition_matrix(labels, n):
    """
    :param labels: 1D array, predicted labels
    :return df_tm: object, transition matrix data frame
    """
    # n = 1 + max(labels)
    tm = [[0] * n for _ in range(n)]
    for (i, j) in zip(labels, labels[1:]):
        tm[i][j] += 1
    B = np.array(tm)
    df_tm = pd.DataFrame(tm)
    B = np.array(tm)
    B_norm = B / B.sum(axis=1)
    return B, df_tm, B_norm


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
