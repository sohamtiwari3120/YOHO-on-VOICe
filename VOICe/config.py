snr = '9dB'
sample_rate = 44100
window_len_secs = 2.56
hop_len_secs = 1.96
max_consecutive_event_silence = 1.0
class_dict = {
    "babycry": 0,
    "gunshot": 1,
    "glassbreak": 2
}
rev_class_dict = ["babycry", "gunshot", "glassbreak"]

mel_hop_len = 441
mel_win_len = 1764
n_fft = 2048
n_mels = 40
fmin = 0
fmax = int(sample_rate/2)

env = 'indoor'
learning_rate = 1e-3
num_classes = 3
num_subwindows = 9
batch_size = 16
# note, we are saving transpose of the logmelspecs
input_height = 257 # hence 257 is actually the length of the time dimension before transpose
input_width = 40 # hence 40 is actually the length of the frequency dimension (n_mels) before transpose
num_workers = 8

# SpecAugment
time_warping_para=5
frequency_masking_para=10
time_masking_para=5
frequency_mask_num=2
time_mask_num=1
# ReduceLRonPlateau
mode = 'min'
patience = 5
factor = 0.5
# trainer params
devices="auto"
accelerator="auto"
gradient_clip_val=0.5

depthwise_layers = [
    # (layer_function, kernel, stride, num_filters)
    ([3, 3], 1,   64),
    ([3, 3], 2,  128),
    ([3, 3], 1,  128),
    ([3, 3], 2,  256),
    ([3, 3], 1,  256),
    ([3, 3], 2,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 1,  512),
    ([3, 3], 2, 1024),
    ([3, 3], 1, 1024),
    ([3, 3], 1, 512),
    ([3, 3], 1, 256),
    ([3, 3], 1, 128),
    # ([3, 3], 1, 128),
    # ([3, 3], 1, 128)
]
