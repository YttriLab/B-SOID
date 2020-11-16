import getopt
import glob
import os
import subprocess
import sys
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

from analysis_subroutines.analysis_utilities.load_data import appdata
from analysis_subroutines.analysis_utilities.processing import data_processing
from analysis_subroutines.analysis_utilities.save_data import results
from analysis_subroutines.analysis_utilities.statistics import rle
from analysis_subroutines.analysis_utilities.visuals import plot_peaks


def get_kinematics(path, name, exp, group_num, bp, fps):
    appdata_ = appdata(path, name)
    _, _, filenames2, data_new, fs_labels = appdata_.load_predictions()
    win_len = np.int(np.round(0.05 / (1 / fps)) * 2 - 1)
    bout_frames = []
    term_frame = []
    pose_all_animal = []
    eu_all_animal = []
    all_bouts_disp = []
    all_bouts_peak_speed = []
    all_bouts_dur = []
    count = 0
    for an, se in enumerate(np.concatenate(exp)):
        bout_frames.append(np.array(np.where(fs_labels[se] == group_num)))
        term_f = np.diff(bout_frames[an]) != 1
        term_frame.append(np.array(term_f*1))
        lengths, pos, grp = rle(term_frame[an].T)
        endpos = np.where(np.diff(pos) < 1)[0][0] + 1
        pos = pos[:endpos]
        poses = data_new[se]
        proc_pose = []
        for col in range(poses.shape[1]):
            pose = data_processing(poses[:, col])
            proc_pose.append(pose.boxcar_center(win_len))
        proc_pose = np.array(proc_pose, dtype=object).T
        pose_all_bp = []
        eu_all_bp = []
        for b in bp:
            pose_single_bp = []
            bt = 0
            eu_dist_bout = []
            for bout in range(0, len(pos) - 1, 2):
                eu_dist_ = []
                pose_single_bp.append(proc_pose[int(bout_frames[an][:, pos[bout]]):
                                                int(bout_frames[an][:, pos[bout+1]])+1,
                                      2 * b:2 * b + 2])
                for row in range(len(pose_single_bp[bt]) - 1):
                    try:
                        eu_dist_ = np.hstack((eu_dist_, np.linalg.norm(pose_single_bp[bt][row + 1, :] -
                                                                       pose_single_bp[bt][row, :])))
                    except TypeError:
                        pass
                eu_dist_ = np.array([np.nan if b_ > np.percentile(eu_dist_, 98) else b_ for b_ in eu_dist_])
                jump_idx = np.where(np.isnan(eu_dist_))[0]
                for ju in jump_idx:
                    try:
                        eu_dist_[ju] = np.nanmean(eu_dist_[ju-1:ju+1])
                    except:
                        pass
                eu_dist_bout.append(eu_dist_)
                bt += 1
            pose_all_bp.append(pose_single_bp)  # all body parts pose estimation in one animal
            eu_all_bp.append(eu_dist_bout)  # all body parts euclidean distance in one animal
        pose_all_animal.append(pose_all_bp)  # all body parts pose estimations for all animals
        eu_all_animal.append(eu_all_bp)  # all body parts euclidean distances for all animals
        bps_bouts_disp = []
        bps_bouts_peak_speed = []
        bps_bouts_dur = []
        for i in tqdm(range(len(eu_all_bp))):
            bouts_disp = []
            bouts_pk_speed = []
            bouts_dur = []
            for j in range(len(eu_all_bp[i])):
                newsig = eu_all_bp[i][j].copy()
                newsig = np.array([0 if a_ < 0.1 * np.max(eu_all_bp[i][j]) else a_ for a_ in newsig])
                pk, info = find_peaks(newsig, prominence=3, distance=(fps / 30))
                try:
                    os.mkdir(str.join('', (path, '/kinematics_analysis')))
                except FileExistsError:
                    pass
                try:
                    os.mkdir(str.join('', (path, '/kinematics_analysis/group{}'.format(group_num))))
                except FileExistsError:
                    pass
                output_path = str.join('', (path, '/kinematics_analysis/group{}'.format(group_num)))
                if pk.size:
                    bout_disp = []
                    for k in range(len(info['left_bases'])):
                        bout_disp.append(np.sum((
                            eu_all_bp[i][j][int(round(info['left_bases'][k])):
                                            int(round(info['right_bases'][k])) + 1])))
                    if pk.size > 3 and np.random.rand() < 0.5:
                        count += 1
                        plot_peaks(eu_all_bp[i][j], None, pk)
                        R = np.linspace(0, 1, len(info['left_bases']))
                        cm = plt.cm.Spectral(R)
                        for k in range(len(info['left_bases'])):
                            plt.fill_between(
                                np.arange(int(round(info['left_bases'][k])),
                                          int(round(info['right_bases'][k])) + 1),
                                eu_all_bp[i][j][int(round(info['left_bases'][k])):
                                                int(round(info['right_bases'][k])) + 1],
                                alpha=0.5, color=cm[k])
                        plt.savefig(output_path + "/file%04d.png" % count)
                        plt.close('all')
                    bouts_disp.append(bout_disp)
                    bouts_pk_speed.append(eu_all_bp[i][j][pk])
                    bouts_dur.append(len(eu_all_bp[i][j]))
            bps_bouts_disp.append(bouts_disp)
            bps_bouts_peak_speed.append(bouts_pk_speed)
            bps_bouts_dur.append(bouts_dur)
        all_bouts_disp.append(bps_bouts_disp)
        all_bouts_peak_speed.append(bps_bouts_peak_speed)
        all_bouts_dur.append(bps_bouts_dur)
    subprocess.call([
        'ffmpeg', '-y', '-framerate', '5', '-i', output_path + '/file%04d.png',
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        output_path + '/kinematics_subsample_examples.mp4'
    ])
    for file_name in glob.glob(output_path + "/*.png"):
        os.remove(file_name)
    return [pose_all_animal, eu_all_animal, all_bouts_disp, all_bouts_peak_speed, all_bouts_dur, output_path]


def group_kinematics(all_bouts_disp, all_bouts_peak_speed, all_bouts_dur, exp):
    bps_exp1_bout_disp = []
    bps_exp2_bout_disp = []
    for j in range(len(all_bouts_disp[0])):
        exp1_bout_disp = []
        exp2_bout_disp = []
        for i in range(len(all_bouts_disp)):
            if any(i == sess for sess in exp[0]):
                if len(all_bouts_disp[i][j]):
                    try:
                        exp1_bout_disp = np.concatenate((exp1_bout_disp, np.concatenate(all_bouts_disp[i][j])))
                    except ValueError:
                        pass

            elif any(i == sess for sess in exp[1]):
                if len(all_bouts_disp[i][j]):
                    try:
                        exp2_bout_disp = np.concatenate((exp2_bout_disp, np.concatenate(all_bouts_disp[i][j])))
                    except ValueError:
                        pass
        bps_exp1_bout_disp.append(exp1_bout_disp)
        bps_exp2_bout_disp.append(exp2_bout_disp)
    bps_exp1_bout_peak_speed = []
    bps_exp2_bout_peak_speed = []
    for j in range(len(all_bouts_peak_speed[0])):
        exp1_bout_peak_speed = []
        exp2_bout_peak_speed = []
        for i in range(len(all_bouts_peak_speed)):
            if any(i == sess for sess in exp[0]):
                if len(all_bouts_peak_speed[i][j]):
                    try:
                        exp1_bout_peak_speed = np.concatenate((exp1_bout_peak_speed,
                                                               np.concatenate(all_bouts_peak_speed[i][j])))
                    except ValueError:
                        pass
            elif any(i == sess for sess in exp[1]):
                if len(all_bouts_peak_speed[i][j]):
                    try:
                        exp2_bout_peak_speed = np.concatenate((exp2_bout_peak_speed,
                                                               np.concatenate(all_bouts_peak_speed[i][j])))
                    except ValueError:
                        pass
        bps_exp1_bout_peak_speed.append(exp1_bout_peak_speed)
        bps_exp2_bout_peak_speed.append(exp2_bout_peak_speed)
    bps_exp1_bout_dur = []
    bps_exp2_bout_dur = []
    for j in range(len(all_bouts_dur[0])):
        exp1_bout_dur = []
        exp2_bout_dur = []
        for i in range(len(all_bouts_dur)):
            if any(i == sess for sess in exp[0]):
                if len(all_bouts_dur[i][j]):
                    try:
                        exp1_bout_dur = np.concatenate((exp1_bout_dur, all_bouts_dur[i][j]))
                    except ValueError:
                        pass
            if any(i == sess for sess in exp[1]):
                if len(all_bouts_dur[i][j]):
                    try:
                        exp2_bout_dur = np.concatenate((exp2_bout_dur, all_bouts_dur[i][j]))
                    except ValueError:
                        pass
        bps_exp1_bout_dur.append(exp1_bout_dur)
        bps_exp2_bout_dur.append(exp2_bout_dur)

    return [bps_exp1_bout_disp, bps_exp2_bout_disp, bps_exp1_bout_peak_speed, bps_exp2_bout_peak_speed,
            bps_exp1_bout_dur, bps_exp2_bout_dur]


def main(argv):
    path = None
    name = None
    group_num = None
    bodyparts = None
    exp = None
    vname = None
    options, args = getopt.getopt(
        argv[1:],
        'p:n:g:b:e:v:',
        ['path=', 'file=', 'group_num=', 'bodyparts=', 'experiment=', 'variable='])
    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-n', '--file'):
            name = option_value
        elif option_key in ('-g', '--group_num'):
            group_num = option_value
        elif option_key in ('-b', '--bodyparts'):
            bodyparts = option_value
        elif option_key in ('-e', '--experiment'):
            exp = option_value
        elif option_key in ('-v', '--variable'):
            vname = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('BEHAVIOR  :', group_num)
    print('BODY PARTS   :', bodyparts)
    print('EXPERIMENT ORDER   :', exp)
    print('VARIABLES   :', vname)
    print('*' * 50)
    print('Computing...')
    [_, _, all_bouts_disp, all_bouts_peak_speed, all_bouts_dur, _] = \
        get_kinematics(path, name, literal_eval(exp), int(group_num), literal_eval(bodyparts), 60)
    [bps_exp1_bout_disp, bps_exp2_bout_disp, bps_exp1_bout_peak_speed, bps_exp2_bout_peak_speed,
     bps_exp1_bout_dur, bps_exp2_bout_dur] = \
        group_kinematics(all_bouts_disp, all_bouts_peak_speed, all_bouts_dur, literal_eval(exp))
    results_ = results(path, name)
    results_.save_sav([bps_exp1_bout_disp, bps_exp2_bout_disp, bps_exp1_bout_peak_speed,
                       bps_exp2_bout_peak_speed, bps_exp1_bout_dur, bps_exp2_bout_dur], vname)


if __name__ == '__main__':
    main(sys.argv)
