import numpy as np
import pandas as pd
from tqdm import tqdm


def read_json_single(filename):
    pose_names = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder",
                  6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip",
                  13: "LKnee", 14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                  20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}
    df = pd.read_json(filename)
    data_arr = df['people']
    data_length = len(data_arr[0]['pose_keypoints_2d'])
    x_val = data_arr[0]['pose_keypoints_2d'][0:data_length:3]
    y_val = data_arr[0]['pose_keypoints_2d'][1:data_length:3]
    l_val = data_arr[0]['pose_keypoints_2d'][2:data_length:3]
    xyl = []
    for i in range(int(data_length / 3)):
        xyl.extend([x_val[i], y_val[i], l_val[i]])
    xyl_array = np.array(xyl).reshape(1, len(xyl))
    a = []
    for i in range(len(pose_names) - 1):
        a.extend((('Openpose', pose_names[i], 'x'), ('OpenPose', pose_names[i], 'y'),
                  ('Openpose', pose_names[i], 'likelihood')))
    micolumns = pd.MultiIndex.from_tuples(a, names=['Algorithm', 'Body parts', 'Frame number'])
    df2 = pd.DataFrame(xyl_array, columns=micolumns)
    fname = filename.rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
    df2.to_csv(str.join('', (filename.rpartition('/')[0], '/', fname, '.csv')), index=True, chunksize=10000,
               encoding='utf-8')
    dfc = pd.read_csv(str.join('', (filename.rpartition('/')[0], '/', fname, '.csv')), low_memory=False)
    return dfc


def json2csv_multi(filenames):
    pose_names = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder",
                  6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip",
                  13: "LKnee", 14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                  20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}
    xyl_array = np.empty((len(filenames), (len(pose_names) - 1) * 3))
    empty_count = 0
    for j, ff in enumerate(tqdm(filenames)):
        df = pd.read_json(ff)
        data_arr = df['people']
        try:
            data_length = len(data_arr[0]['pose_keypoints_2d'])
            x_val = data_arr[0]['pose_keypoints_2d'][0:data_length:3]
            y_val = data_arr[0]['pose_keypoints_2d'][1:data_length:3]
            l_val = data_arr[0]['pose_keypoints_2d'][2:data_length:3]
            xyl = []
            for i in range(int(data_length / 3)):
                xyl.extend([x_val[i], y_val[i], l_val[i]])
            xyl_array[j, :] = np.array(xyl).reshape(1, len(xyl))
        except KeyError:
            xyl_array[j, :] = xyl_array[j - 1, :]
            empty_count += 1
    a = []
    for i in range(len(pose_names) - 1):
        a.extend((('Openpose', pose_names[i], 'x'), ('OpenPose', pose_names[i], 'y'),
                  ('Openpose', pose_names[i], 'likelihood')))
    micolumns = pd.MultiIndex.from_tuples(a, names=['Algorithm', 'Body parts', 'Frame number'])
    df2 = pd.DataFrame(xyl_array, columns=micolumns)
    fname = filenames[0].rpartition('/')[-1].rpartition('_')[0].rpartition('_')[0]
    df2.to_csv(str.join('', (filenames[0].rpartition('/')[0], '/', fname, '.csv')), index=True, chunksize=10000,
               encoding='utf-8')
    return
