import re
from operator import itemgetter

import numpy as np
import pandas as pd


def convert_int(s):
    """ Converts digit string to integer
    """
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)


def reorganize_group_order(accuracy_data, order):
    accuracy_ordered = []
    for i in range(len(accuracy_data)):
        accuracy_ordered.append(itemgetter(order)(accuracy_data[i]))
    return accuracy_ordered


class data_processing:

    def __init__(self, data):
        self.data = data

    def boxcar_center(self, n):
        a1 = pd.Series(self.data)
        return np.array(a1.rolling(window=n, min_periods=1, center=True).mean())
