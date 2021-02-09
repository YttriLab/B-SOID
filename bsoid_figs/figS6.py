import sys
import subprocess


print('\n \n \n A2A CASPASE KINEMATICS ANALYSIS \n \n \n')
path = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/workspace/a2a_loc_Rhkin.mat'
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/kinematics_cdf/'
print('\n DATA FROM {} \n'.format(path))
print('-' * 50)


# FIGS6
print('\n' * 1)
print('Preparing head groom kinematics cdf curves as xxx.{} found in fig6b...'.format(fig_format))
print('\n' * 1)
variables = str(['c_loc_len_N4_1', 'a_loc_len_N4_1'])
c = str(['deepskyblue', 'red'])
x_range = str([1, 7])
leg = '0'

p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf.py',
                      '-p', path, '-v', variables,
                      '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()


variables = str(['c_loc_speed_N4_1', 'a_loc_speed_N4_1'])
x_range = str([10, 50])
leg = '0'

p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf.py',
                      '-p', path, '-v', variables,
                      '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()


variables = str(['c_loc_dur_N4', 'a_loc_dur_N4'])
x_range = str([0, 4])


p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf.py',
                      '-p', path, '-v', variables,
                      '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
p.communicate()
p.kill()


# FIGS6
# print('\n' * 1)
# print('Preparing face groom kinematics cdf curves as xxx.{} found in fig6d...'.format(fig_format))
# print('\n' * 1)
# variables = str(['c_it_len_N4_1', 'a_it_len_N4_1'])
# c = str(['deepskyblue', 'red'])
# x_range = str([1, 5])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf.py',
#                       '-p', path, '-v', variables,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# variables = str(['c_it_speed_N4_1', 'a_it_speed_N4_1'])
# x_range = str([7, 35])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf.py',
#                       '-p', path, '-v', variables,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# variables = str(['c_it_dur_N4', 'a_it_dur_N4'])
# x_range = str([0, 2])
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf.py',
#                       '-p', path, '-v', variables,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()

