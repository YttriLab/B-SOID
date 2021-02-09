import sys
import subprocess
import os


print('\n \n \n HUMAN EXERCISE UMAP \n \n \n')
path = '/Volumes/Elements/exercise_data/output/'
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/human_exercise/'
print('\n DATA FROM {} \n'.format(path))
print('-' * 50)


# FIG S2
print('\n' * 1)
print('Preparing UMAP + HDBSCAN xxx.{} found in fig S2...'.format(fig_format))
print('\n' * 1)
name = 'exercise_30fps_n2fixed'

p = subprocess.Popen([sys.executable, './subroutines/umap_clustering_plot.py',
                      '-p', path, '-f', name, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()


# FIG S2
vidpath = '/Volumes/Elements/exercise_data/'
mp4name = 'TEST/mp4s/'
projectname = 'eric_exercise_fixed/'
for g in range(20):
    for e in range(10):
        mp4file = 'group_{}_example_{}.mp4'.format(g, e)
        if os.path.exists(str.join('', (vidpath, mp4name, projectname, mp4file))):
            var = str.join('', (mp4file.partition('.')[0], '_images'))

            p = subprocess.Popen([sys.executable, './subroutines/extract_images.py',
                                  '-p', vidpath, '-n', mp4name, '-f', projectname,
                                  '-g', mp4file, '-v', var])
            p.communicate()
            p.kill()

# FIG S2
vidpath = '/Volumes/Elements/exercise_data/'
mp4name = 'TRAIN/mp4s/'
projectname = 'Andy_exercise2_fixed/'
for g in range(20):
    for e in range(10):
        mp4file = 'group_{}_example_{}.mp4'.format(g, e)
        if os.path.exists(str.join('', (vidpath, mp4name, projectname, mp4file))):
            var = str.join('', (mp4file.partition('.')[0], '_images'))

            p = subprocess.Popen([sys.executable, './subroutines/extract_images.py',
                                  '-p', vidpath, '-n', mp4name, '-f', projectname,
                                  '-g', mp4file, '-v', var])
            p.communicate()
            p.kill()

