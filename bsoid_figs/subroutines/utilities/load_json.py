import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

path = '/Volumes/Elements/Drive/Data/Nahom/output/exercise2/'
POSE_BODY_25_BODY_PARTS = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder", 6: "LElbow",
                           7: "LWrist", 8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee",
                           14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe", 20: "LSmallToe",
                           21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}
filenames = glob.glob(path + '/*.json')
xyl_array = np.empty((len(filenames), 75))
empty_count = 0
for j, f in enumerate(tqdm(filenames)):
    df = pd.read_json(f)
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
    except IndexError:
        xyl_array[j, :] = xyl_array[j - 1, :]
        empty_count += 1
empty_ratio = empty_count / j
print('There are {}% empty jsons'.format(empty_ratio * 100))
a = []
for i in range(len(POSE_BODY_25_BODY_PARTS)-1):
    a.extend((('Openpose', POSE_BODY_25_BODY_PARTS[i], 'x'), ('OpenPose', POSE_BODY_25_BODY_PARTS[i], 'y'),
             ('Openpose', POSE_BODY_25_BODY_PARTS[i], 'likelihood')))
micolumns = pd.MultiIndex.from_tuples(a, names=['Algorithm', 'Body parts', 'Frame number'])
df = pd.DataFrame(xyl_array, columns=micolumns)
df.to_csv(str.join('', (path, 'exercise2.csv')), index=True, chunksize=10000, encoding='utf-8')
