import csv
from config import sample_rate, window_len_secs, hop_len_secs, class_dict, mel_hop_len, mel_win_len, n_fft, n_mels, fmax, fmin, num_subwindows
import soundfile as sf
import math
import numpy as np
import librosa


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
