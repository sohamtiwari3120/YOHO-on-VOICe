from specAugment import spec_augment_pytorch
import csv
import glob
import numpy as np
from typing import List
import soundfile as sf
import math
import numpy as np
import librosa
import os
from torch.utils.data import Dataset, DataLoader
from subprocess import Popen, PIPE
from config import sample_rate, window_len_secs, hop_len_secs, class_dict, mel_hop_len, mel_win_len, n_fft, n_mels, fmax, fmin, num_subwindows, snr, time_warping_para, frequency_masking_para, time_masking_para, frequency_mask_num, time_mask_num
from tqdm import tqdm
from types import file_paths_type

envs = ['vehicle', 'outdoor', 'indoor']
data_mode = ['training', 'test', 'validation']

file_paths: file_paths_type = {}


for mode in data_mode:
    file_paths[mode] = {}
    base_path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'data', snr)
    for e in envs:
        file_paths[mode][e] = [base_path + p[0]
                               for p in read_annotation(os.path.join(base_path, f"{e}_source_{mode}.txt"))]


def read_annotation(filepath):
    """Reads and returns the annotations in filepath

    Args:
        filepath (str): path of the file containing annotations

    Returns:
        List[]: list of rows in the file
    """
    events = []
    with open(filepath, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            events.append(row)
    return events


def convert_to_mono():
    """Convert audios to mono channels.
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(os.path.join(base_dir, f'{snr}-mono'), exist_ok=True)

    training_files = [ele for sublist in list(
        file_paths['training'].values()) for ele in sublist]
    test_files = [ele for sublist in list(
        file_paths['test'].values()) for ele in sublist]
    validation_files = [ele for sublist in list(
        file_paths['validation'].values()) for ele in sublist]

    # conversion to mono
    for sound in tqdm(training_files+test_files+validation_files):
        temp_file = convert_path_to_mono(sound)
        command = command = "sox " + sound + " " + temp_file + " channels 1"
        p = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        output, err = p.communicate()

    # adding subfolders in place
    for mode in data_mode:
        for e in envs:
            os.makedirs(os.path.join(
                base_dir, f'{snr}-mono', f"{mode}-data", f"{e}"), exist_ok=True)


def construct_audio_windows(audio_path, sample_rate=sample_rate, window_len_secs=window_len_secs, hop_len_secs=hop_len_secs):
    """chunks audio files into windows with hop_len, and returns these chunked audio files as well as the start and end time of each window.

    Args:
        audio_path (str): path of audio file
        sample_rate (int, optional): Sampling Rate at which to sample audios . Defaults to sample_rate.
        window_len_secs (float, optional): Length of audio windows in seconds. Later converted to number of samples using sampling_rate. Defaults to window_len_secs.
        hop_len_secs (float, optional): Hop length in seconds. Later converted to number of samples using sampling_rate. Defaults to hop_len_secs.

    Returns:
        List: list of np arrays of length window_len
        List[Tuple[float, float]]: list of tuples of start_time and end_time of corresponding windows
    """
    win_len = int(sample_rate*window_len_secs)
    hop_len = int(sample_rate*hop_len_secs)

    a, _ = sf.read(audio_path, samplerate=sample_rate)

    if a.shape[0] < win_len:
        a_padded = np.zeros((win_len, ))
        a_padded[0:a.shape[0]] = a

    else:
        no_of_hops = math.ceil((a.shape[0] - win_len) / hop_len)
        a_padded = np.zeros((int(win_len + hop_len*no_of_hops), ))
        a_padded[0:a.shape[0]] = a

    a_ex = [a_padded[i - win_len: i]
            for i in range(win_len, a_padded.shape[0]+1, hop_len)]
    win_ranges = [((i - win_len)/sample_rate, i/sample_rate)
                  for i in range(win_len, a_padded.shape[0]+1, hop_len)]

    # chunks large audio file into windows, and gives the start and end time of each window
    return a_ex, win_ranges


def extract_anns_for_audio_window(annotation_path, window_start_secs, window_end_secs, window_len_secs=window_len_secs):
    """Given the annotation file, returns the annotations corresponding to the audio window in focus.

    Args:
        annotation_path (str): Path of annotation file.
        window_start_secs (float): Start time of particular audio window in seconds.
        window_end_secs (float): End time of particular audio window in seconds.
        window_len_secs (Numeric): Duration of the audio window in seconds.
    """
    events = read_annotation(annotation_path)
    ann = [[float(e[0]), float(e[1]), e[2]] for e in events]

    curr_ann = []

    for a in ann:
        #   window_start_secs = 1
        #   window_end_secs = 2
        #   a = [1.1, 1.2, baby]
        #   a = [1.9, 2.2, baby]
        #   a = [0.9, 2.2, baby]
        if a[1] > window_start_secs and a[0] < window_end_secs:  # checking if it exceeds the window
            # if a[0] >= window_start_secs and a[0] < window_end_secs: => this will check only if the audio starts within the window
            curr_start = max(a[0] - window_start_secs, 0.0)
            curr_end = min(a[1] - window_start_secs, window_len_secs)
            curr_ann.append([curr_start, curr_end, a[2]])

    # obtaining the list of all unique anns in the current focus window
    class_wise_events = {}
    for c in curr_ann:
        if c[2] in class_wise_events:
            class_wise_events[c[2]].append(c)
        else:
            class_wise_events[c[2]] = [c]

        # grouping all annotations by their class
        # {
        #     "baby":[
        #             [1.1, 1.3, 'baby'],
        #             [1.2, 1.4, 'baby'],
        #             [0.1, 1.2, 'baby'],
        #             [1.1, 2.2, 'baby'],
        #     ],
        #     "gun":[
        #             [1.1, 1.2, 'baby'],
        #             [0.1, 1.2, 'baby'],
        #             [1.1, 2.2, 'baby'],
        #     ],
        #     ....
        # }

    max_event_silence = 0.0
    all_events = []
    for k in list(class_wise_events.keys()):
        curr_events = class_wise_events[k]
        count = 0
        # skipping the last ann in that class to compare ann[i] and ann[i+1]
        while count < len(curr_events) - 1:
            if (curr_events[count][1] >= curr_events[count + 1][0]) or (curr_events[count + 1][0] - curr_events[count][1] <= max_event_silence):
                # merging two annotations for the same time period into 1
                curr_events[count][1] = max(
                    curr_events[count + 1][1], curr_events[count][1])
                del curr_events[count + 1]
            else:
                count += 1

        all_events += curr_events
        # all events is corrected dictionary in the form of 2d list, removing distinc
        #     all_events = [
        #             [1.1, 1.3, 'baby'],
        #             [1.2, 1.4, 'baby'],
        #             [0.1, 1.2, 'baby'],
        #             [1.1, 2.2, 'baby'],
        #                   ...
        #             [1.1, 1.2, 'gun'],
        #             [0.1, 1.2, 'gun'],
        #             [1.1, 2.2, 'gun'],
        #                   ...
        #             [1.1, 1.2, 'breaking'],
        #             [0.1, 1.2, 'breaking'],
        #             [1.1, 2.2, 'breaking'],
        #     ],
    for i in range(len(all_events)):
        all_events[i][0] = round(all_events[i][0], 3)
        all_events[i][1] = round(all_events[i][1], 3)

    all_events.sort(key=lambda x: x[0])
#   sorted all events by their start time, so can be possible that ann -> baby, gun,
    return all_events


def get_model_compatible_anns(events, window_len_secs=window_len_secs, num_subwindows=num_subwindows):
    """Converts the annotations for a particular audio window/spectrogram into format appropriate for the output layer of the model. Also normalizes the event start and end times to a value between 0 and 1 for the audio window in focus.

    Args:
        events (List): List of annotations for particular audio window
        window_len_secs (float, optional): Length of audio window. Defaults to 10.0.
        num_subwindows (int, optional): Number of subwindows to divide the audio window in. Defaults to 9.

    Returns:
        numpy.array: Shape (num_subwindows, len(class_dict.keys()) * 3). Normalized, Yoho compatible output.
    """
    #   its generating output formatted for for neural network
    # REMEMBER: bins are disjoint sequences, frames can be overlapping
    bin_length = window_len_secs/num_subwindows
    labels = np.zeros((num_subwindows, len(class_dict.keys()) * 3))

    for e in events:

        start_time = float(e[0])
        stop_time = float(e[1])

        start_bin = int(start_time // bin_length)
        stop_bin = int(stop_time // bin_length)

        start_time_2 = start_time - start_bin * bin_length
        stop_time_2 = stop_time - stop_bin * bin_length

        n_bins = stop_bin - start_bin

        if n_bins == 0:
            labels[start_bin, class_dict[e[2]] * 3:class_dict[e[2]]
                   * 3 + 3] = [1, start_time_2, stop_time_2]

        elif n_bins == 1:
            labels[start_bin, class_dict[e[2]] * 3:class_dict[e[2]]
                   * 3 + 3] = [1, start_time_2, bin_length]

            if stop_time_2 > 0.0:
                labels[stop_bin, class_dict[e[2]] * 3:class_dict[e[2]]
                       * 3 + 3] = [1, 0.0, stop_time_2]

        elif n_bins > 1:
            labels[start_bin, class_dict[e[2]] * 3:class_dict[e[2]]
                   * 3 + 3] = [1, start_time_2, bin_length]

            for i in range(1, n_bins):
                labels[start_bin + i, class_dict[e[2]] *
                       3:class_dict[e[2]] * 3 + 3] = [1, 0.0, bin_length]

            if stop_time_2 > 0.0:
                labels[stop_bin, class_dict[e[2]] * 3:class_dict[e[2]]
                       * 3 + 3] = [1, 0.0, stop_time_2]

    # labels[:, [1, 2, 4, 5]] /= bin_length => normalising values

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if j % 3 != 0:
                labels[i][j] /= bin_length

    return labels


def get_log_melspectrogram(audio, sr=sample_rate, hop_length=mel_hop_len, win_length=mel_win_len, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax):
    """Return the log-scaled Mel bands of an audio signal."""
    audio_2 = librosa.util.normalize(audio)
    bands = librosa.feature.melspectrogram(
        y=audio_2, sr=sr, hop_length=hop_length, win_length=win_length, n_fft=n_fft, n_mels=n_mels)
    return librosa.core.power_to_db(bands)


def convert_path_to_mono(path):
    """To convert input file/folder path to mono path

    Args:
        path (str): Input file path

    Returns:
        str: Output file path for mono files
    """
    return path.replace(f"{snr}", f"{snr}-mono")


def generate_windows_and_anns(mode: str, env: str, sample_rate=sample_rate, window_len_secs=window_len_secs, hop_len_secs=hop_len_secs, num_subwindows=num_subwindows):
    """Function to generate and return audio windows and corresponding model compatible annotations/labels

    Args:
        mode (str): one of ['training', 'test', 'validation']
        env (str): one of envs in config
        sample_rate (int, optional): Sample rate. Defaults to sample_rate.
        window_len_secs (_type_, optional): Audio window - window_length in seconds. Defaults to window_len_secs.
        hop_len_secs (_type_, optional): Audio window - hop_length in seconds. Defaults to hop_len_secs.
        num_subwindows (int, optional): Number of subwindows to divide audio window in. Defaults to num_subwindows.

    Raises:
        Exception: If invalid environment type chosen.
        Exception: If invalid data mode

    Returns:
        List[numpy.array]: list of all numpy arrays for the audio windows, all concatenated into one list.
        List[numpy.array]: list of all model compatible annotations for corresponding audio windows.

    """
    if env not in envs:
        raise Exception('Invalid environment type.')
    if mode not in data_mode:
        raise Exception('Invalid data mode.')
    audio_windows = []
    labels = []

    for i, audio_path in enumerate(file_paths[mode][env]):
        audio_path = convert_path_to_mono(audio_path)
        audio_wins, window_ranges = construct_audio_windows(
            audio_path, sample_rate, window_len_secs, hop_len_secs)
        audio_windows.extend(audio_wins)

        ann_path = file_paths[mode][env][i].replace('.wav', '.txt')
        for w in window_ranges:
            anns = extract_anns_for_audio_window(
                ann_path, w[0], w[1], window_len_secs)
            compatible_ann = get_model_compatible_anns(
                anns, window_len_secs, num_subwindows)
            labels.append(compatible_ann)

    return audio_windows, labels


def save_logmelspec_and_labels(mode, env, audio_windows, labels, snr=snr):
    """To save the generated logmelspecs and compatible annotations in npy format.

    Args:
        mode (str): Type/mode of data being processed.
        env (str): Environment of data.
        audio_windows (List[numpy.array]): audio windows returned from generate_windows_and_anns function.
        labels (List[numpy.array]): model compatible annotations/labels returned from generate_windows_and_anns function.
        snr (str, optional): Audio snr level. Defaults to snr.

    Raises:
        Exception: If invalid environment type chosen.
        Exception: If invalid data mode
    """
    if env not in envs:
        raise Exception('Invalid environment type.')
    if mode not in data_mode:
        raise Exception('Invalid data mode.')

    base_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'data')
    folder_path = os.path.join(
        base_dir, f'{snr}-mono', f'{mode}-data', env)
    logmel_path = os.path.join(folder_path, 'logmels_npy')
    label_path = os.path.join(folder_path, 'labels_npy')
    os.makedirs(logmel_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    for i, (audio_win, label) in enumerate(zip(audio_windows, labels)):
        logmelspec = get_log_melspectrogram(audio_win).T
        np.save(os.path.join(logmel_path, f'logmelspec-{i}.npy'), logmelspec)
        np.save(os.path.join(label_path, f'label-{i}.npy'), label)

    return logmel_path, label_path


class VOICeDataset(Dataset):
    """Custom PyTorch Dataset class for VOICe dataset.
    """

    def __init__(self, mode: str, logmel_path: str, label_path: str, spec_transform=False):
        """Initialises the VOICe dataset class to load data for given mode and env. (the logmel_path and label_path variables are env and mode specific.)

        Args:
            mode (str): One of ['training', 'test', 'validation']
            logmel_path (str): Path to folder containing the saved logmel npy files
            label_path (str): Path to folder containing the saved compatible annotations/label npy files.
            spec_transform (bool, optional): SpecAugmentation for spectrograms performed if true. Defaults to False.

        Raises:
            Exception: If invalid data mode chosen.
        """
        if mode not in data_mode:
            raise Exception('Invalid data mode.')
        self.mode = mode
        self.logmel_path = logmel_path
        self.label_path = label_path
        self.spec_transform = spec_transform

        self.logmel_npy = glob.glob(self.logmel_path+f'logmelspec-*.npy')
        self.label_npy = glob.glob(self.label_path+f'label-*.npy')

    def __len__(self):
        return len(self.logmel_npy)

    def __getitem__(self, idx):
        X = np.load(self.logmel_npy[idx])[:, None]
        y = np.load(self.label_npy[idx])

        if self.spec_transform and self.mode == 'training':
            X = spec_augment_pytorch.spec_augment(X, time_warping_para=time_warping_para, frequency_masking_para=frequency_masking_para,
                                                  time_masking_para=time_masking_para, frequency_mask_num=frequency_mask_num, time_mask_num=time_mask_num)
        return X, y
