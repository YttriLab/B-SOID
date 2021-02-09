import numpy as np
from utilities.load_data import appdata
from kfold_accuracy import reorganize_accuracy
from utilities.save_data import results
import sys, getopt
from ast import literal_eval

def generate_coherence(path, name, fps, target_fps, frame_skips, animal_index, t, order):
    appdata_ = appdata(path, name)
    flders, flder, filenames, data_new, fs_labels = appdata_.load_predictions()
    coherence_data = []
    labels = np.repeat(fs_labels[animal_index], np.floor(target_fps / fps))
    t = int(t * np.floor(target_fps / fps))
    for i in frame_skips:
        downsampled_labels = labels[0:t:i]
        filled_labels = np.repeat(downsampled_labels, i)
        coh_vec = []
        for j in range(len(np.unique(fs_labels[0][0:t]))):
            coh_vec.append(
                len(np.argwhere((filled_labels[0:t] - labels[0:t] == 0) & (labels[0:t] == j)))
                / len(np.argwhere(labels[0:t] == j)))
        coherence_data.append(np.array(coh_vec))
    coherence_data = np.array(coherence_data)
    coherence_reordered = reorganize_accuracy(coherence_data, order)
    return np.array(coherence_reordered)



def main(argv):
    path = None
    name = None
    fps = None
    target_fps = None
    frame_skips = None
    animal_index = None
    t = None
    order = None
    vname = None
    options, args = getopt.getopt(
        argv[1:],
        'p:n:f:F:s:i:t:o:v:',
        ['path=', 'file=', 'fps=', 'target_fps=', 'frame_skips=', 'animal_idx=', 'time=', 'order=', 'variable='])

    for option_key, option_value in options:
        if option_key in ('-p', '--path'):
            path = option_value
        elif option_key in ('-n', '--file'):
            name = option_value
        elif option_key in ('-f', '--framerate'):
            fps = option_value
        elif option_key in ('-F', '--target_fps'):
            target_fps = option_value
        elif option_key in ('-s', '--frame_skips'):
            frame_skips = option_value
        elif option_key in ('-i', '--animal_idx'):
            animal_index = option_value
        elif option_key in ('-t', '--time'):
            t = option_value
        elif option_key in ('-o', '--order'):
            order = option_value
        elif option_key in ('-v', '--variable'):
            vname = option_value
    print('*' * 50)
    print('PATH   :', path)
    print('NAME   :', name)
    print('FRAMERATE  :', fps)
    print('TARGET FRAMERATE   :', target_fps)
    print('FRAME SKIPS   :', frame_skips)
    print('ANIMAL INDEX   :', animal_index)
    print('TIME   :', t)
    print('ORDER    :', order)
    print('VARIABLE   :', vname)
    print('*' * 50)
    print('Computing...')
    coherence_reordered = generate_coherence(path, name, int(fps), int(target_fps), literal_eval(frame_skips),
                                             int(animal_index), int(t), literal_eval(order))
    results_ = results(path, name)
    results_.save_sav(coherence_reordered, vname)


if __name__ == '__main__':
    main(sys.argv)

