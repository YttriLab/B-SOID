"""
likelihood processing analysis_utilities
Forward fill low likelihood (x,y)
"""

import glob
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def boxcar_center(a, n):
    a1 = pd.Series(a)
    moving_avg = np.array(a1.rolling(window=n, min_periods=1, center=True).mean())

    return moving_avg


def convert_int(s):
    if s.isdigit():
        return int(s)
    else:
        return s


def alphanum_key(s):
    return [convert_int(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    l.sort(key=alphanum_key)


def get_filenames(base_path, folder):
    filenames = glob.glob(base_path + folder + '/*.csv')
    sort_nicely(filenames)
    return filenames


def get_filenamesh5(base_path, folder):
    filenames = glob.glob(base_path + folder + '/*.h5')
    sort_nicely(filenames)
    return filenames


def get_filenamesjson(base_path, folder):
    filenames = glob.glob(base_path + folder + '/*.json')
    sort_nicely(filenames)
    return filenames


def import_folders(base_path, folders: list, pose):
    fldrs = []
    filenames = []
    rawdata_li = []
    data_li = []
    perc_rect_li = []
    for i, fd in enumerate(folders):  # Loop through folders
        f = get_filenames(base_path, fd)
        for j, filename in enumerate(f):
            curr_df = pd.read_csv(filename, low_memory=False)
            curr_df_filt, perc_rect = adp_filt(curr_df, pose)
            rawdata_li.append(curr_df)
            perc_rect_li.append(perc_rect)
            data_li.append(curr_df_filt)
        fldrs.append(fd)
        filenames.append(f)
    data = np.array(data_li)
    return fldrs, filenames, data, perc_rect_li


def adp_filt(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    currdf = np.array(currdf[1:])
    for header in pose:
        if currdf[0][header + 1] == "likelihood":
            lIndex.append(header)
        elif currdf[0][header + 1] == "x":
            xIndex.append(header)
        elif currdf[0][header + 1] == "y":
            yIndex.append(header)
    curr_df1 = currdf[:, 1:]
    datax = curr_df1[1:, np.array(xIndex)]
    datay = curr_df1[1:, np.array(yIndex)]
    data_lh = curr_df1[1:, np.array(lIndex)]
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        data_lh_float = data_lh[:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect


def adp_filt_h5(currdf: object, pose):
    lIndex = []
    xIndex = []
    yIndex = []
    headers = np.array(currdf.columns.get_level_values(2)[:])
    for header in pose:
        if headers[header] == "likelihood":
            lIndex.append(header)
        elif headers[header] == "x":
            xIndex.append(header)
        elif headers[header] == "y":
            yIndex.append(header)
    curr_df1 = np.array(currdf)
    datax = curr_df1[:, np.array(xIndex)]
    datay = curr_df1[:, np.array(yIndex)]
    data_lh = curr_df1[:, np.array(lIndex)]
    currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
    perc_rect = []
    for i in range(data_lh.shape[1]):
        perc_rect.append(0)
    for x in tqdm(range(data_lh.shape[1])):
        a, b = np.histogram(data_lh[1:, x].astype(np.float))
        rise_a = np.where(np.diff(a) >= 0)
        if rise_a[0][0] > 1:
            llh = b[rise_a[0][0]]
        else:
            llh = b[rise_a[0][1]]
        data_lh_float = data_lh[:, x].astype(np.float)
        perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[0, x], datay[0, x]])
        for i in range(1, data_lh.shape[0]):
            if data_lh_float[i] < llh:
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[i, x], datay[i, x]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect


def adp_filt_sleap_h5(currdf: object, pose):
    datax = currdf['tracks'][0][0][pose]
    datay = currdf['tracks'][0][1][pose]
    currdf_filt = np.zeros((datax.shape[1], (datax.shape[0]) * 2))
    perc_rect = []
    for i in range(len(pose)):
        perc_rect.append(np.argwhere(np.isnan(datax[i]) == True).shape[0] / datax.shape[1])
    for x in tqdm(range(datax.shape[0])):
        first_not_nan = np.where(np.isnan(datax[x, :]) == False)[0][0]
        currdf_filt[0, (2 * x):(2 * x + 2)] = np.hstack([datax[x, first_not_nan], datay[x, first_not_nan]])
        for i in range(1, datax.shape[1]):
            if np.isnan(datax[x][i]):
                currdf_filt[i, (2 * x):(2 * x + 2)] = currdf_filt[i - 1, (2 * x):(2 * x + 2)]
            else:
                currdf_filt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[x, i], datay[x, i]])
    currdf_filt = np.array(currdf_filt)
    currdf_filt = currdf_filt.astype(np.float)
    return currdf_filt, perc_rect


def no_filt_sleap_h5(currdf: object, pose):
    datax = currdf['tracks'][0][0][pose]
    datay = currdf['tracks'][0][1][pose]
    pose_ = []
    currdf_nofilt = np.zeros((datax.shape[1], (datax.shape[0]) * 2))
    for x in tqdm(range(datax.shape[0])):
        pose_.append(currdf['node_names'][pose[x]])
        print(pose_)
        for i in range(0, datax.shape[1]):
            currdf_nofilt[i, (2 * x):(2 * x + 2)] = np.hstack([datax[x, i], datay[x, i]])
    currdf_nofilt = np.array(currdf_nofilt)
    header = pd.MultiIndex.from_product([['SLEAP'],
                                         [i for i in pose_],
                                         ['x', 'y']],
                                        names=['algorithm', 'pose', 'coord'])
    df = pd.DataFrame(currdf_nofilt, columns=header)
    return df
