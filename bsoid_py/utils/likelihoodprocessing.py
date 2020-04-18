"""
likelihood processing utilities
Forward fill low likelihood (x,y)
"""
from bsoid_py.config import *
from bsoid_py.utils.visuals import *
import glob
import pandas as pd
import re
from tqdm import tqdm

def boxcar_center(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg


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


def get_filenames(folder):
    """
    Gets a list of filenames within a folder
    :param folder: str, folder path
    :return: list, filenames
    """
    filenames = glob.glob(BASE_PATH + folder + '/*.csv')
    sort_nicely(filenames)
    return filenames


def import_folders(folders: list):
    """
    Import multiple folders containing .csv files and process them
    :param folders: list of folder paths
    :return: dict, keys=filenames, values=processed image arrays (bw)
    """
    filenames = []
    rawdata_li = []
    data_li = []
    perc_rect_li = []
    for i, fd in enumerate(folders):  # Loop through folders
        f = get_filenames(fd)
        for j, filename in enumerate(f):
            logging.info('Importing CSV file {} from folder {}'.format(j + 1, i + 1))
            currDf = pd.read_csv(filename, low_memory=False)
            currDf_filt, perc_rect = adp_filt(currDf)
            logging.info('Done preprocessing (x,y) from file {}, folder {}.'.format(j + 1, i + 1))
            rawdata_li.append(currDf)
            perc_rect_li.append(perc_rect)
            data_li.append(currDf_filt)
        filenames.append(f)
        logging.info('Processed {} CSV files from folder: {}'.format(len(f), fd))
    data = np.array(data_li)
    logging.info('Processed a total of {} CSV files, and compiled into a {} data list.'.format(len(data_li),
                                                                                               data.shape))
    return filenames, data, perc_rect_li


def adp_filt(currDf):
    lIndex = []
    xIndex = []
    yIndex = []
    currDf = np.array(currDf[1:])
    for header in range(len(currDf[0])):
        if currDf[0][header] == "likelihood":
            lIndex.append(header)
        elif currDf[0][header] == "x":
            xIndex.append(header)
        elif currDf[0][header] == "y":
            yIndex.append(header)
    logging.info('Extracting likelihood value...')
    currDf = np.array(currDf)
    currDf1 = currDf[:, 1:]
    datax = currDf1[:, np.array(xIndex) - 1]
    datay = currDf1[:, np.array(yIndex) - 1]
    data_lh = currDf1[:, np.array(lIndex) - 1]
    currDf_filt = np.zeros((datax.shape[0] - 1, (datax.shape[1]) * 2))
    perc_rect = []
    logging.info('Computing data threshold to forward fill any sub-threshold (x,y)...')
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = (b[rise_a[0][0]+1])
        else:
            llh = (b[rise_a[0][1]+1])
        data_lh_float = data_lh[1:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        for i in range(1, data_lh.shape[0] - 1):
            if data_lh_float[i] < llh:
                currDf_filt[i, (2 * x):(2 * x + 2)] = currDf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currDf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currDf_filt = np.array(currDf_filt[1:])
    currDf_filt = currDf_filt.astype(np.float)
    return currDf_filt, perc_rect


def main(folders):
    filenames, data, perc_rect = import_folders(folders)
    return filenames, data, perc_rect


if __name__ == '__main__':
    main(TRAIN_FOLDERS)
