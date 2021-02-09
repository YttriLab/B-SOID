import sys
import subprocess


print('\n \n \n B-SOID QUANTIFICATION \n \n \n')
path = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/workspace/MvsBvsS_zscore_mse3.mat'
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/motion_energy/'
print('\n DATA FROM {} \n'.format(path))
print('-' * 50)


# FIG5B
print('\n' * 1)
print('Preparing image MSE comparison matrix as xxx.{} found in fig5b...'.format(fig_format))
print('\n' * 1)
algorithm = 'MotionMapper'
c = 'Orange'
c_range = str([0, 3])
cline = 'deepskyblue'
n = '4'
colorbar = '0'
p = subprocess.Popen([sys.executable, './subroutines/immse_heatmap.py',
                      '-p', path, '-a', algorithm,
                      '-c', c, '-r', c_range, '-n', n, '-l', cline, '-m', fig_format, '-o', outpath, '-b', colorbar])
p.communicate()
p.kill()

algorithm = 'B-SOiD'
cline = 'hotpink'

p = subprocess.Popen([sys.executable, './subroutines/immse_heatmap.py',
                      '-p', path, '-a', algorithm,
                      '-c', c, '-r', c_range, '-n', n, '-l', cline, '-m', fig_format, '-o', outpath, '-b', colorbar])
p.communicate()
p.kill()

# FIG5C
print('\n' * 1)
print('Preparing image MSE cdf curves as xxx.{} found in fig5c...'.format(fig_format))
print('\n' * 1)
# c = str(['deepskyblue', 'hotpink', 'gray'])
c = str(['deepskyblue', 'hotpink', 'black'])
x_range = str([0, 3])
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/motion_energy/'

p = subprocess.Popen([sys.executable, './subroutines/immse_cdf.py',
                      '-p', path,
                      '-c', c, '-r', x_range, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()
print("All xxx.{}s generated. see {} for results".format(fig_format, outpath))
print('-' * 50)