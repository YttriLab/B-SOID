import os
import ffmpeg
import cv2
from utilities.save_data import results
from utilities.processing import sort_nicely
import sys, getopt


def get_images(pathname, group_num):
    try:
        os.mkdir(str.join('', (pathname, '/pngs')))
    except FileExistsError:
        pass
    try:
        os.mkdir(str.join('', (pathname, 'pngs', '/', group_num)))
    except FileExistsError:
        pass
    frame_dir = str.join('', (pathname, 'pngs', '/', group_num))
    print('You have created {} as your PNG directory for video {}.'.format(frame_dir, group_num))
    probe = ffmpeg.probe(os.path.join(pathname, group_num))
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    num_frames = int(video_info['nb_frames'])
    bit_rate = int(video_info['bit_rate'])
    avg_frame_rate = round(
        int(video_info['avg_frame_rate'].rpartition('/')[0]) / int(video_info['avg_frame_rate'].rpartition('/')[2]))
    print('Extracting {} frames at {} frames per second...'.format(num_frames, avg_frame_rate))
    try:
        (ffmpeg.input(os.path.join(pathname, group_num))
         .filter('fps', fps=avg_frame_rate)
         .output(str.join('', (frame_dir, '/frame%01d.png')), video_bitrate=bit_rate,
                 s=str.join('', (str(int(width)), 'x', str(int(height)))), sws_flags='bilinear',
                 start_number=0)
         .run(capture_stdout=True, capture_stderr=True))
        print('Done extracting {} frames from video {}.'.format(num_frames, group_num))
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
    image_files = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
    sort_nicely(image_files)
    im = []
    for i in range(len(image_files)):
        im.append(cv2.imread(os.path.join(frame_dir, image_files[i])))
    return im, image_files


def main(argv):
    path = None
    mp4name = None
    projectname = None
    group_num = None
    var = None
    options, args = getopt.getopt(
        argv[1:],
        'p:n:f:g:v:',
        ['path=', 'mp4_name=', 'project_name=',  'group_num=', 'variable='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-n', '--mp4_name'):
            mp4name = option_value
        elif option_key in ('-f', '--project_name'):
            projectname = option_value
        elif option_key in ('-g', '--group_num'):
            group_num = option_value
        elif option_key in ('-v', '--variable'):
            var = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('MP4 NAME   :', mp4name)
    print('PROJECT   :', projectname)
    print('BEHAVIOR  :', group_num)
    print('VARIABLES   :', var)
    print('*' * 50)
    print('Computing...')
    pathname = str.join('', (path, mp4name, projectname))
    im, image_files = get_images(pathname, group_num)
    results_ = results(pathname, '')
    results_.save_sav([im, image_files], var)


if __name__ == '__main__':
    main(sys.argv)
