import numpy as np
from numpy import trapz
import os
from scipy.signal import find_peaks, resample, peak_widths
from utilities.detect_peaks import _plot
import matplotlib.pyplot as plt
from tqdm import tqdm
from utilities.load_data import appdata
from utilities.save_data import results
from utilities.statistics import rle
from utilities.processing import data_processing
import sys, getopt
from ast import literal_eval


def get_kinematics(path, name, group_num, bp, FPS):
    appdata_ = appdata(path, name)
    _, _, filenames2, data_new, fs_labels = appdata_.load_predictions()
    win_len = np.int(np.round(0.05 / (1 / FPS)) * 2 - 1)
    bout_frames = []
    term_frame = []
    pose_all_animal = []
    eu_all_animal = []
    all_bouts_disp = []
    all_bouts_peak_speed = []
    all_bouts_dur = []
    for an in range(len(data_new)):
        bout_frames.append(np.array(np.where(fs_labels[an] == group_num)))
        term_f = np.diff(bout_frames[an]) != 1
        term_frame.append(np.array(term_f*1))
        lengths, pos, grp = rle(term_frame[an].T)
        endpos = np.where(np.diff(pos) < 1)[0][0] + 1
        pos = pos[:endpos]
        poses = data_new[an]
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
                    eu_dist_ = np.hstack((eu_dist_, np.linalg.norm(pose_single_bp[bt][row + 1, :] -
                                                                   pose_single_bp[bt][row, :])))
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
                newsig = np.array([0 if a_ < 0.05 * np.max(eu_all_bp[i][j]) else a_ for a_ in newsig])
                pk, info = find_peaks(newsig, prominence=2, distance=int(FPS/10))  # prominence 2 was better than 1
                try:
                    os.mkdir('/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/'
                             'kinematics_analysis/{}'.format(name))
                except FileExistsError:
                    pass
                try:
                    os.mkdir('/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/'
                             'kinematics_analysis/{}/behavior{}'.format(name, group_num))
                except FileExistsError:
                    pass
                output_path = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/' \
                              'kinematics_analysis/{}/group{}/file{}'.format(name, group_num, an)
                if pk.size:
                    bout_disp = []
                    for k in range(len(info['left_bases'])):
                        bout_disp.append(np.sum((
                            eu_all_bp[i][j][int(round(info['left_bases'][k])):int(round(info['right_bases'][k]))])))
                    if pk.size > 3:
                        _plot(eu_all_bp[i][j], None, pk)
                        R = np.linspace(0, 1, len(info['left_bases']))
                        cm = plt.cm.Spectral(R)
                        for k in range(len(info['left_bases'])):
                            plt.fill_between(
                                np.arange(int(round(info['left_bases'][k])),
                                          int(round(info['right_bases'][k]))),
                                eu_all_bp[i][j][int(round(info['left_bases'][k])):int(round(info['right_bases'][k]))],
                                alpha=0.5, color=cm[k])
                        plt.savefig(str.join('', (output_path, 'pose{}_bout{}_kinematics_extraction.'.format(i, j),
                                                  'png')), format='png', transparent=True)
                        plt.close('all')
                    bouts_disp.append(np.array(bout_disp, dtype=object))
                    bouts_pk_speed.append(eu_all_bp[i][j][pk])
                    bouts_dur.append(len(eu_all_bp[i][j]))
                # else:
                    # bouts_disp.append(np.array(0, dtype=object))
                    # bouts_pk_speed.append(np.array(0, dtype=object))
                    # bouts_dur.append(np.array(0, dtype=object))
            bps_bouts_disp.append(np.array(bouts_disp, dtype=object))
            bps_bouts_peak_speed.append(np.array(bouts_pk_speed, dtype=object))
            bps_bouts_dur.append(np.array(bouts_dur, dtype=object))
        all_bouts_disp.append(np.array(bps_bouts_disp, dtype=object))
        all_bouts_peak_speed.append(np.array(bps_bouts_peak_speed, dtype=object))
        all_bouts_dur.append(np.array(bps_bouts_dur, dtype=object))

    return pose_all_animal, eu_all_animal, all_bouts_disp, all_bouts_peak_speed, all_bouts_dur


def group_kinematics(all_bouts_disp, all_bouts_peak_speed, all_bouts_dur, exp):
    bps_exp1_bout_disp = []
    bps_exp2_bout_disp = []
    for j in range(len(all_bouts_disp[0])):
        exp1_bout_disp = []
        exp2_bout_disp = []
        for i in range(len(all_bouts_disp)):
            if any(i == sess for sess in exp[0]):
                for j in range(len(all_bouts_disp[i])):
                    if all_bouts_disp[i][j].size:
                        try:
                            exp1_bout_disp = np.concatenate((exp1_bout_disp, all_bouts_disp[i][j]))
                        except ValueError:
                            pass

            elif any(i == sess for sess in exp[1]):
                for j in range(len(all_bouts_disp[i])):
                    if all_bouts_disp[i][j].size:
                        try:
                            exp2_bout_disp = np.concatenate((exp2_bout_disp, all_bouts_disp[i][j]))
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
                for j in range(len(all_bouts_peak_speed[i])):
                    if all_bouts_peak_speed[i][j].size:
                        try:
                            exp1_bout_peak_speed = np.concatenate((exp1_bout_peak_speed, all_bouts_peak_speed[i][j]))
                        except ValueError:
                            pass

            elif any(i == sess for sess in exp[1]):
                for j in range(len(all_bouts_peak_speed[i])):
                    if all_bouts_peak_speed[i][j].size:
                        try:
                            exp2_bout_peak_speed = np.concatenate((exp2_bout_peak_speed, all_bouts_peak_speed[i][j]))
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
                for j in range(len(all_bouts_dur[i])):
                    if all_bouts_dur[i][j].size:
                        try:
                            exp1_bout_dur = np.concatenate((exp1_bout_dur, all_bouts_dur[i][j]))
                        except ValueError:
                            pass

            if any(i == sess for sess in exp[1]):
                for j in range(len(all_bouts_dur[i])):
                    if all_bouts_dur[i][j].size:
                        try:
                            exp2_bout_dur = np.concatenate((exp2_bout_dur, all_bouts_dur[i][j]))
                        except ValueError:
                            pass
        bps_exp1_bout_dur.append(exp1_bout_dur)
        bps_exp2_bout_dur.append(exp2_bout_dur)

    return bps_exp1_bout_disp, bps_exp2_bout_disp, bps_exp1_bout_peak_speed, bps_exp2_bout_peak_speed, \
           bps_exp1_bout_dur, bps_exp2_bout_dur


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
    _, _, all_bouts_disp, all_bouts_peak_speed, all_bouts_dur = \
        get_kinematics(path, name, int(group_num), literal_eval(bodyparts), 60)
    bps_exp1_bout_disp, bps_exp2_bout_disp, bps_exp1_bout_peak_speed, bps_exp2_bout_peak_speed, \
    bps_exp1_bout_dur, bps_exp2_bout_dur = group_kinematics(all_bouts_disp, all_bouts_peak_speed, all_bouts_dur,
                                                            literal_eval(exp))
    results_ = results(path, name)
    results_.save_sav([bps_exp1_bout_disp, bps_exp2_bout_disp, bps_exp1_bout_peak_speed, bps_exp2_bout_peak_speed, \
    bps_exp1_bout_dur, bps_exp2_bout_dur], vname)


if __name__ == '__main__':
    main(sys.argv)
