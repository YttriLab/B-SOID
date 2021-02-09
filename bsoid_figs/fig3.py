import sys
import subprocess


print('\n \n \n B-SOID QUANTIFICATION \n \n \n')
path = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/workspace/l5neural5ms_.mat'
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/neural_data/'
print('\n DATA FROM {} \n'.format(path))
print('-' * 50)

# FIG3A
print('\n' * 1)
print('Preparing neural matrix as xxx.{} found in fig3a...'.format(fig_format))
print('\n' * 1)
algorithm = 'non-frameshifted'
c = 'Orange'
c_range = str([0, 3])
cline = 'black'
n = '4'
colorbar = '0'
p = subprocess.Popen([sys.executable, './subroutines/neural_plot.py',
                      '-p', path, '-a', algorithm,
                      '-c', c, '-r', c_range, '-n', n, '-l', cline, '-m', fig_format, '-o', outpath, '-b', colorbar])
p.communicate()
p.kill()

algorithm = 'frameshifted'

p = subprocess.Popen([sys.executable, './subroutines/neural_plot.py',
                      '-p', path, '-a', algorithm,
                      '-c', c, '-r', c_range, '-n', n, '-l', cline, '-m', fig_format, '-o', outpath, '-b', colorbar])
p.communicate()
p.kill()