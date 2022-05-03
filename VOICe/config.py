from traceback import print_tb
from dotenv import load_dotenv
load_dotenv() #loading environment variables
import os
import sys

def add_EAP_to_path():
    """Function to add EAP - External-Attention-pytorch to the module's path.
    """    
    EAP_PATH = os.getenv('EAP_PATH')
    if EAP_PATH is not None:
        sys.path.insert(0, EAP_PATH)
        return 1
    else:
        print('Error EAP_PATH not found in .env file.')
        return 0
        
class hparams:
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

    # process_dataset
    save_logmelspec: bool = False
    save_labels: bool = True

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
    batch_size = 64
    # note, we are saving transpose of the logmelspecs
    # hence 257 is actually the length of the time dimension before transpose
    input_height = 257
    # hence 40 is actually the length of the frequency dimension (n_mels) before transpose
    input_width = n_mels
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

    models = ['Yoho', 'VOICeConvNeXt', 'VOICePANN', 'VOICeViT', 'VOICeCoAtNet', 'VOICeConvMixer', 'VOICePANNYoho']
    model_name = models[0]

    # evaluate
    source_env = 'indoor'
    target_env = 'vehicle'
    data_mode = 'test'

    # kervolutional
    kernel_modes = ['linear', 'polynomial', 'gaussian']

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
    min_delta = 0.1
    tf_patience = 5
    tf_monitor = 'val_loss'

    # PYTORCH
    # Pytorch SpecAugment
    time_warping_para = 5
    frequency_masking_para = 10
    time_masking_para = 5
    frequency_mask_num = 2
    time_mask_num = 1

    # Pytorch ReduceLRonPlateau
    mode = 'min'
    patience = 5
    factor = 0.5
    monitor = 'validation_loss'

    # PL trainer params
    devices = "auto"
    accelerator = "auto"
    gradient_clip_val = 0.5
    loss_function_str: str = 'weighted_mse'

    # Adam Optimiser
    adam_eps = 1e-7
    adam_weight_decay = 0

    # layer initializer
    initialize_layer: bool = True

    # PyTorch Dataloader
    train_shuffle = True
    val_shuffle = False
    test_shuffle = False

    # PyTorch VOICe Dataset
    train_spec_transform = False
    val_spec_transform = False
    test_spec_transform = False

    # PANN
    pann_encoder_ckpt_path_cnn10 = f'/notebooks/YOHO/YOHO-on-VOICe/VOICe/models/Cnn10_mAP0.380.pth'
    pann_encoder_ckpt_path_cnn14 = f'/notebooks/YOHO/YOHO-on-VOICe/VOICe/models/Cnn14_mAP=0.431.pth'
    pann_versions = ["Cnn10", "Cnn14"]
    pann_version = pann_versions[1]
    # CNN 10
    output_embedding: bool = False

    # CBAM
    use_cbam = True
    cbam_channels = 64
    cbam_reduction_factor = 4
    cbam_kernel_size = 7

    # ViT
    output_1d_embeddings = False
    patch_size=8
    dim=1024
    depth=6
    heads=16
    mlp_dim=2048
    vit_dropout=0.1
    vit_emb_dropout=0.

class YOHO_hparams(hparams):
    # (NHWC) (-1, 129, 20, 32)
    use_cbam = False
    cbam_channels = 32
    cbam_reduction_factor = 2
    cbam_kernel_size = 3

    # Patchify
    use_patches = False

    # UFO Attention
    use_ufo = True
    ufo_d_k=512
    ufo_d_v=512
    ufo_h=8

    # ParNet Attention Usage
    use_pna = True
    pna_channels = 32

    # MobileViT Attention Usage
    use_mva = True
    mva_in_channel=128
    mva_dim=512
    mva_kernel_size=3
    mva_patch_size=5


class CoAtNet_hparams(hparams):
    # (NHWC) (-1, 129, 20, 32)
    use_cbam = True
    cbam_channels = 32
    cbam_reduction_factor = 2
    cbam_kernel_size = 3