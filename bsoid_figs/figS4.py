import sys
import subprocess


print('\n \n \n FRAMSHIFT NEURAL DIFFERNCES \n \n \n')
path = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/workspace/neuralbehavior_durs.mat'
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/neural_data/'
print('\n DATA FROM {} \n'.format(path))
print('-' * 50)


# FIG6B
print('\n' * 1)
print('Preparing head groom kinematics cdf curves as xxx.{} found in fig6b...'.format(fig_format))
print('\n' * 1)
variables = str(['L5neuralbehavioral'])
c = str(['black', 'magenta'])
x_range = str([0, 1.5])
order = str([4, 5, 7, 0, 3, 2, 1, 6, 8, 9, 10])

p = subprocess.Popen([sys.executable, './subroutines/fsdiff_hist.py',
                      '-p', path, '-v', variables,
                      '-c', c, '-r', x_range, '-O', order, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()