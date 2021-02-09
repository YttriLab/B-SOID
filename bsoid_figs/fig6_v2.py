import sys
import subprocess


print('\n \n \n KINEMATICS ANALYSIS \n \n \n')
path = '/Volumes/Elements/B-SOID/output3/'
fig_format = 'png'
outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/kinematics_py/'
print('\n DATA FROM {} \n'.format(path))
print('-' * 50)


print('\n' * 1)
print('Preparing kinematics cdf as xxx.{} found on GitHub...'.format(fig_format))
print('\n' * 1)
name = 'openfield_60min_N6'
group_num = '10'
body_parts = str([0, 1, 2, 3, 4, 5])
exp = str([[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]])
var = 'locRf_kin_C57_N6'

p = subprocess.Popen([sys.executable, './subroutines/extract_kinematics.py',
                      '-p', path, '-n', name,
                      '-g', group_num, '-b', body_parts, '-e', exp, '-v', var])
p.communicate()
p.kill()

#
# vname = 'Distance'
# bp = '1'
# c = str(['deepskyblue', 'red'])
# x_range = str([0.5, 6.5])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
# vname = 'Speed'
# bp = '1'
# x_range = str([5, 45])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
# vname = 'Duration'
# bp = '1'
# x_range = str([0, 2])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# print('\n \n \n A2A CASPASE KINEMATICS ANALYSIS \n \n \n')
# path = '/Volumes/Elements/B-SOID/output3/'
# fig_format = 'png'
# outpath = '/Volumes/Elements/Manuscripts/B-SOiD/bsoid_natcomm/figure_panels/kinematics_py/'
# print('\n DATA FROM {} \n'.format(path))
# print('-' * 50)
#
#
# print('\n' * 1)
# print('Preparing kinematics cdf as xxx.{} found on GitHub...'.format(fig_format))
# print('\n' * 1)
# name = 'a2a_60min_N4'
# group_num = '10'
# body_parts = str([0, 1])
# exp = str([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
# var = 'locRf_kin_A2ACaspase_N4'
#
# p = subprocess.Popen([sys.executable, './subroutines/extract_kinematics.py',
#                       '-p', path, '-n', name,
#                       '-g', group_num, '-b', body_parts, '-e', exp, '-v', var])
# p.communicate()
# p.kill()
#
#
# vname = 'Distance'
# bp = '1'
# c = str(['deepskyblue', 'red'])
# x_range = str([0.5, 6.5])
# leg = '1'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Speed'
# bp = '1'
# x_range = str([5, 45])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Duration'
# bp = '1'
# x_range = str([0, 2])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
#
# group_num = '3'
# body_parts = str([0, 1])
# exp = str([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
# var = 'facegrRf_kin_A2ACaspase_N4'
#
# p = subprocess.Popen([sys.executable, './subroutines/extract_kinematics.py',
#                       '-p', path, '-n', name,
#                       '-g', group_num, '-b', body_parts, '-e', exp, '-v', var])
# p.communicate()
# p.kill()
#
#
# vname = 'Distance'
# bp = '1'
# x_range = str([0, 4])
# leg = '1'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Speed'
# bp = '1'
# x_range = str([5, 15])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Duration'
# bp = '1'
# x_range = str([0, 4])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
#
#
# group_num = '2'
# body_parts = str([0, 1])
# exp = str([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
# var = 'headgrRf_kin_A2ACaspase_N4'
#
# p = subprocess.Popen([sys.executable, './subroutines/extract_kinematics.py',
#                       '-p', path, '-n', name,
#                       '-g', group_num, '-b', body_parts, '-e', exp, '-v', var])
# p.communicate()
# p.kill()
#
#
# vname = 'Distance'
# bp = '1'
# x_range = str([0, 4])
# leg = '1'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Speed'
# bp = '1'
# x_range = str([5, 15])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Duration'
# bp = '1'
# x_range = str([0, 4])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# group_num = '6'
# body_parts = str([0, 1, 2, 3, 4, 5])
# exp = str([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]])
# var = 'itchRf_kin_A2ACaspase_N4'
#
# p = subprocess.Popen([sys.executable, './subroutines/extract_kinematics.py',
#                       '-p', path, '-n', name,
#                       '-g', group_num, '-b', body_parts, '-e', exp, '-v', var])
# p.communicate()
# p.kill()
#
#
# vname = 'Distance'
# bp = '3'
# x_range = str([0.5, 4.5])
# leg = '1'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Speed'
# bp = '3'
# x_range = str([5, 37])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()
#
#
# vname = 'Duration'
# bp = '3'
# x_range = str([0, 2])
# leg = '0'
#
# p = subprocess.Popen([sys.executable, './subroutines/kinematics_cdf_v2.py',
#                       '-p', path, '-n', name, '-v', var, '-V', vname, '-b', bp,
#                       '-c', c, '-r', x_range, '-l', leg, '-m', fig_format, '-o', outpath])
# p.communicate()
# p.kill()