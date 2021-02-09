import numpy as np
import pandas as pd
import re


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


class data_processing:

    def __init__(self, data):
        self.data = data

    def boxcar_center(self, n):
        a1 = pd.Series(self.data)
        moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())
        return moving_avg