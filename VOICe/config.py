snr = '9dB'
sample_rate = 44100
window_len_secs = 2.56
hop_len_secs = 1.96
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

num_classes = 3
learning_rate = 1e-3
num_subwindows = 9
batch_size = 32

# SpecAugment
time_warping_para=80
frequency_masking_para=8
time_masking_para=25
frequency_mask_num=1
time_mask_num=2

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
