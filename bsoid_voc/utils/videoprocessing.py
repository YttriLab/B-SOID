"""
Extracting frames from videos
"""

import glob
import random

import numpy as np
import cv2
from tqdm import tqdm

from bsoid_voc.utils.likelihoodprocessing import sort_nicely
from bsoid_voc.utils.visuals import *


def get_vidnames(folder):
    """
    Gets a list of filenames within a folder
    :param folder: str, folder path
    :return: list, video filenames
    """
    vidnames = glob.glob(BASE_PATH + folder + '/*.mp4')
    sort_nicely(vidnames)
    return vidnames


def vid2frame(vidname, labels, fps, output_path=FRAME_DIR):
    """
    Extracts frames every 100ms to match the labels for visualizations
    :param vidname: string, path to video
    :param labels: 1D array, labels from training
    :param fps: scalar, frame-rate of original camera
    :param output_path: string, path to output
    """
    vidobj = cv2.VideoCapture(vidname)
    pbar = tqdm(total=int(vidobj.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = vidobj.get(3)
    height = vidobj.get(4)
    labels = np.hstack((labels[0],labels)) # fill the first frame
    count = 0
    count1 = 0
    font_scale = 1
    font = cv2.FONT_HERSHEY_COMPLEX
    rectangle_bgr = (0, 0, 0)
    while vidobj.isOpened():
        ret, frame = vidobj.read()
        if ret:
            text = 'Group' + str(labels[count1])
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
            text_offset_x = 50
            text_offset_y = 50
            box_coords = ((text_offset_x - 12, text_offset_y + 12),
                          (text_offset_x + text_width + 12, text_offset_y - text_height - 8))
            cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(frame, text, (text_offset_x, text_offset_y), font,
                        fontScale=font_scale, color=(255, 255, 255), thickness=1)
            cv2.imwrite(os.path.join(output_path, 'frame{:d}.png'.format(count1)), frame)
            count += round(fps / 10)  # i.e. at 60fps, this skips every 5
            count1 += 1
            vidobj.set(1, count)
            pbar.update(round(fps / 10))
        else:
            vidobj.release()
            break
    pbar.close()
    return


def import_vidfolders(folders: list, output_path: list):
    """
    Import multiple folders containing .mp4 files and extract frames from them
    :param folders: list of folder paths
    :param output_path: list, directory to where you want to store extracted vid images in LOCAL_CONFIG
    """
    vidnames = []
    for i, fd in enumerate(folders):  # Loop through folders
        v = get_vidnames(fd)
        for j, vidname in enumerate(v):
            logging.info('Extracting frames from {} and appending labels to these images...'.format(vidname))
            vid2frame(vidname, output_path)
            logging.info('Done extracting images and writing labels, from MP4 file {}'.format(j + 1))
        vidnames.append(v)
        logging.info('Processed {} MP4 files from folder: {}'.format(len(v), fd))
    return


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


def create_labeled_vid(labels, crit=3, counts=3, frame_dir=FRAME_DIR, output_path=SHORTVID_DIR):
    """
    :param labels: 1D array, labels from training or testing
    :param crit: scalar, minimum duration for random selection of behaviors, default 300ms
    :param counts: scalar, number of randomly generated examples, default 5
    :param frame_dir: string, directory to where you extracted vid images in LOCAL_CONFIG
    :param output_path: string, directory to where you want to store short video examples in LOCAL_CONFIG
    """
    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(frame_dir, images[0]))
    height, width, layers = frame.shape
    rnges = []
    n, idx, lengths = repeatingNumbers(labels)
    idx2 = []
    for i, j in enumerate(lengths):
        if j >= crit:
            rnges.append(range(idx[i], idx[i] + j))
            idx2.append(i)
    for i in tqdm(range(0, len(np.unique(labels)))):
        a = []
        for j in range(0, len(rnges)):
            if n[idx2[j]] == i:
                a.append(rnges[j])
        try:
            rand_rnges = random.sample(a, counts)
            for k in range(0, len(rand_rnges)):
                video_name = 'group_{}_example_{}.mp4'.format(i, k)
                grpimages = []
                for l in rand_rnges[k]:
                    grpimages.append(images[l])
                video = cv2.VideoWriter(os.path.join(output_path, video_name), fourcc, 5, (width, height))
                for image in grpimages:
                    video.write(cv2.imread(os.path.join(frame_dir, image)))
                cv2.destroyAllWindows()
                video.release()
        except:
            pass
    return


def main(vidname, labels, fps, output_path):
    vid2frame(vidname, labels, fps, output_path)
    create_labeled_vid(labels, crit=3, counts=5, frame_dir=output_path, output_path=SHORTVID_DIR)
    return

