import sys
import subprocess
import os

print('\n \n \n MODEL PERFORMANCE \n \n \n')
path = '/Volumes/Elements/B-SOID/output3/'
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/model_performance/'
print('\n DATA FROM {} \n'.format(path))
print('-' * 50)


# FIG2A
vidpath = '/Volumes/Elements/B-SOID/datasets/'
mp4name = '080219/mp4s/'
projectname = '2019-08-02_10-56-50cut30min_1hrDeepCut_resnet50_OpenFieldHighResApr8shuffle1_1030000/'
for g in range(11):
    for e in range(5):
        mp4file = 'group_{}_example_{}.mp4'.format(g, e)
        if os.path.exists(str.join('', (vidpath, mp4name, projectname, mp4file))):
            var = str.join('', (mp4file.partition('.')[0], '_images'))

            p = subprocess.Popen([sys.executable, './subroutines/extract_images.py',
                                  '-p', vidpath, '-n', mp4name, '-f', projectname,
                                  '-g', mp4file, '-v', var])
            p.communicate()
            p.kill()


# FIG2B
print('\n' * 1)
print('Preparing confusion matrix heatmap as xxx.{} found in fig2b...'.format(fig_format))
print('\n' * 1)


# FIG2C
print('\n' * 1)
print('Preparing accuracy boxplot as xxx.{} found in fig2c...'.format(fig_format))
print('\n' * 1)
name = 'openfield_60min_N6'
k = '10'
order = str([4, 5, 7, 0, 3, 2, 1, 6, 8, 9, 10])
var = 'accuracy_kf'
algorithm = 'Randomforests'
c = str(['indianred', 'indianred',
         'goldenrod', 'goldenrod',
         'royalblue', 'royalblue', 'royalblue', 'royalblue',
         'mediumseagreen', 'mediumseagreen', 'mediumseagreen'])

# p = subprocess.Popen([sys.executable, './subroutines/kfold_accuracy.py',
#                       '-p', path, '-f', name,
#                       '-o', order, '-k', k, '-v', var])
# p.communicate()
# p.kill()

p = subprocess.Popen([sys.executable, './subroutines/accuracy_boxplot.py',
                      '-p', path, '-f', name, '-v', var,
                      '-a', algorithm, '-c', c, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()

# FIG2D/G
print('\n' * 1)
print('Preparing limb trajectories as xxx.{} found in fig2d/g right...'.format(fig_format))
print('\n' * 1)

animal_index = '0'
bodyparts = str([1, 2, 3, 4])
time_range = str([(20*60+42) * 60 - 1, (20*60+44) * 60]) # headgroom 2 scratch
order1 = str([0, 2])
order2 = str([1, 3])
c = str(['coral', 'cyan'])
p = subprocess.Popen([sys.executable, './subroutines/trajectory_plot.py',
                      '-p', path, '-f', name, '-i', animal_index, '-b', bodyparts, '-t', time_range,
                      '-r', order1, '-R', order2, '-c', c, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()

print('\n' * 1)
print('Preparing limb trajectories as xxx.{} found in fig2d/g left...'.format(fig_format))
print('\n' * 1)

time_range = str([int((11*60+19.5)*60-1), int((11*60+21.5)*60)])
p = subprocess.Popen([sys.executable, './subroutines/trajectory_plot.py',
                      '-p', path, '-f', name, '-i', animal_index, '-b', bodyparts, '-t', time_range,
                      '-r', order1, '-R', order2, '-c', c, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()

# FIG2F
print('\n' * 1)
print('Preparing coherence boxplot as xxx.{} found in fig2f...'.format(fig_format))
print('\n' * 1)

name = 'openfield_200fps'
fps = '200'
target_fps = '600'
frame_skips = str([60, 30, 12, 6, 4])
animal_index = '0'
# order = str([4, 5, 7, 0, 3, 1, 6, 8, 9, 10])
order = str([4, 5, 7, 0, 1, 6, 8, 9, 10])
time = '300000'
var = 'coherence_data'

p = subprocess.Popen([sys.executable, './subroutines/frameshift_coherence.py',
                      '-p', path, '-n', name, '-f', fps, '-F', target_fps, '-s', frame_skips,
                      '-i', animal_index, '-o', order, '-t', time, '-v', var])
p.communicate()
p.kill()

algorithm = 'Randomforests'
c = 'k'

p = subprocess.Popen([sys.executable, './subroutines/coherence_boxplot.py',
                      '-p', path, '-f', name, '-v', var,
                      '-a', algorithm, '-c', c, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()

print("All xxx.{}s generated. see {} for results".format(fig_format, outpath))
print('-' * 50)