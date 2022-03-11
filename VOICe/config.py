# dataset metadata
class_dict = {
    "babycry": 0,
    "gunshot": 1,
    "glassbreak": 2
}
rev_class_dict = list(class_dict.keys())
num_classes = len(rev_class_dict)

# loading audiofiles
snr = '9dB'
sample_rate = 44100
window_len_secs = 2.56
hop_len_secs = 1.96
max_consecutive_event_silence = 0.3
num_subwindows = 9

# dataset melspec
mel_hop_len = 441
mel_win_len = 1764
n_fft = 2048
n_mels = 40
fmin = 0
fmax = int(sample_rate/2)

# common framework agnostic model params
env = 'indoor'
learning_rate = 1e-3
batch_size = 16
# note, we are saving transpose of the logmelspecs
input_height = 257 # hence 257 is actually the length of the time dimension before transpose
input_width = 40 # hence 40 is actually the length of the frequency dimension (n_mels) before transpose
num_workers = 8

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

backends = ['pytorch', 'tensor_flow']
backend = backends[0]

# TENSORFLOW
# tensorflow regularizers
l2_kernel_reg_first_conv2d = 1e-3
l2_bias_reg_first_conv2d = 1e-3
l2_kernel_reg_remaining_conv2d = 1e-2
l2_bias_reg_remaining_conv2d = 1e-2

# tensorflow datagenerator
shuffle = False

# tensorflow batchnorm
batch_norm_eps = 1e-4

# tensorflow spatial dropout
spatial_dropout = 0.1

# tensorflow epochs 
epochs = 1000

# tensorflow model.fit verbose
fit_verbose = 1

# tensorflow earlystopping callback
min_delta=0.1
tf_patience=5
tf_monitor='val_loss'

# PYTORCH
# Pytorch SpecAugment
time_warping_para=5
frequency_masking_para=10
time_masking_para=5
frequency_mask_num=2
time_mask_num=1

# Pytorch ReduceLRonPlateau
mode = 'min'
patience = 5
factor = 0.5
monitor='validation_loss'

# PL trainer params
devices="auto"
accelerator="auto"
gradient_clip_val=0.5
loss_function_str: str = 'weighted_mse'

# Adam Optimiser
adam_eps = 1e-7

# layer initializer
initialize_layer: bool = True