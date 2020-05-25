"""
Extracting frames from videos
"""

import glob
import random

import numpy as np
import cv2
import ffmpeg
from tqdm import tqdm

from bsoid_app.utils.likelihoodprocessing import sort_nicely
from bsoid_app.utils.visuals import *


def repeatingNumbers(labels):
    """
    :param labels: 1D array, predicted labels
    :return n_list: 1D array, the label number
    :return idx: 1D array, label start index
    :return lengths: 1D array, how long each bout lasted for
    """
    i = 0
    n_list = []
    idx = []
    lengths = []
    while i < len(labels) - 1:
        n = labels[i]
        n_list.append(n)
        startIndex = i
        idx.append(i)
        while i < len(labels) - 1 and labels[i] == labels[i + 1]:
            i = i + 1
        endIndex = i
        length = endIndex - startIndex
        lengths.append(length)
        i = i + 1
    return n_list, idx, lengths


def create_labeled_vid(labels, crit, counts, output_fps, frame_dir, output_path):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    rnges = []
    n, idx, lengths = repeatingNumbers(labels)
    idx2 = []
    for i, j in enumerate(lengths):
        if j >= crit:
            rnges.append(range(idx[i], idx[i] + j))
            idx2.append(i)
    for b, i in enumerate(tqdm(np.unique(labels))):
        a = []
        for j in range(0, len(rnges)):
            if n[idx2[j]] == i:
                a.append(rnges[j])
        try:
            rand_rnges = random.sample(a, min(len(a), counts))
            for k in range(0, len(rand_rnges)):
                video_name = 'group_{}_example_{}.mp4'.format(i, k)
                grpimages = []
                for l in rand_rnges[k]:
                    grpimages.append(images[l])
                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, output_fps, (width, height))
                for image in grpimages:
                    video.write(cv2.imread(os.path.join(frame_dir, image)))
                cv2.destroyAllWindows()
                video.release()
        except:
            pass
    return



