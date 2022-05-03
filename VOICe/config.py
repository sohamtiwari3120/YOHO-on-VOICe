from mimetypes import init
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
    def __init__(self) -> None:

        self.seed = 0
        # dataset metadata
        self.class_dict = {
            "babycry": 0,
            "gunshot": 1,
            "glassbreak": 2
        }
        self.rev_class_dict = list(self.class_dict.keys())
        self.num_classes = len(self.rev_class_dict)
    
        # loading audiofiles
        self.snr = '9dB'
        self.sample_rate = 44100
        self.window_len_secs = 2.56
        self.hop_len_secs = 1.96
        self.max_consecutive_event_silence = 0.3
        self.num_subwindows = 9
    
        # process_dataset
        self.save_logmelspec: bool = False
        self.save_labels: bool = True
    
        # dataset melspec
        self.mel_hop_len = 441
        self.mel_win_len = 1764
        self.n_fft = 2048
        self.n_mels = 40
        self.fmin = 0
        self.fmax = int(self.sample_rate/2)
    
        # common framework agnostic model params
        self.env = 'indoor'
        self.learning_rate = 1e-3
        self.batch_size = 64
        # note, we are saving transpose of the logmelspecs
        # hence 257 is actually the length of the time dimension before transpose
        self.input_height = 257
        # hence 40 is actually the length of the frequency dimension (n_mels) before transpose
        self.input_width = self.n_mels
        self.num_workers = 8
    
        self.depthwise_layers = [
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
    
        self.backends = ['pytorch', 'tensor_flow']
        self.backend = self.backends[0]
    
        self.models = ['Yoho', 'VOICeConvNeXt', 'VOICePANN', 'VOICeViT', 'VOICeCoAtNet', 'VOICeConvMixer', 'VOICePANNYoho']
        self.model_name = self.models[0]
    
        # evaluate
        self.source_env = 'indoor'
        self.data_mode = 'test'
        self.envs = ['vehicle', 'outdoor', 'indoor']
        self.target_envs = self.envs
        self.data_modes = ['training', 'test', 'validation']
        # kervolutional
        self.kernel_modes = ['linear', 'polynomial', 'gaussian']
    
        # TENSORFLOW
        # tensorflow regularizers
        self.l2_kernel_reg_first_conv2d = 1e-3
        self.l2_bias_reg_first_conv2d = 1e-3
        self.l2_kernel_reg_remaining_conv2d = 1e-2
        self.l2_bias_reg_remaining_conv2d = 1e-2
    
        # tensorflow datagenerator
        self.shuffle = False
    
        # tensorflow batchnorm
        self.batch_norm_eps = 1e-4
    
        # tensorflow spatial dropout
        self.spatial_dropout = 0.1
    
        # tensorflow epochs
        self.epochs = 1000
    
        # tensorflow model.fit verbose
        self.fit_verbose = 1
    
        # tensorflow earlystopping callback
        self.min_delta = 0.1
        self.tf_patience = 5
        self.tf_monitor = 'val_loss'
    
        # PYTORCH
        # Pytorch SpecAugment
        self.time_warping_para = 5
        self.frequency_masking_para = 10
        self.time_masking_para = 5
        self.frequency_mask_num = 2
        self.time_mask_num = 1
    
        # Pytorch ReduceLRonPlateau
        self.mode = 'min'
        self.patience = 5
        self.factor = 0.5
        self.monitor = 'validation_loss'
    
        # PL trainer params
        self.devices = "auto"
        self.accelerator = "auto"
        self.gradient_clip_val = 0.5
        self.loss_function_str: str = 'weighted_mse'
    
        # Adam Optimiser
        self.adam_eps = 1e-7
        self.adam_weight_decay = 0
    
        # layer initializer
        self.initialize_layer: bool = True
    
        # PyTorch Dataloader
        self.train_shuffle = True
        self.val_shuffle = False
        self.test_shuffle = False
    
        # PyTorch VOICe Dataset
        self.train_spec_transform = False
        self.val_spec_transform = False
        self.test_spec_transform = False
    
        # PANN
        self.pann_encoder_ckpt_path_cnn10 = f'/notebooks/YOHO/YOHO-on-VOICe/VOICe/models/Cnn10_mAP0.380.pth'
        self.pann_encoder_ckpt_path_cnn14 = f'/notebooks/YOHO/YOHO-on-VOICe/VOICe/models/Cnn14_mAP=0.431.pth'
        self.pann_versions = ["Cnn10", "Cnn14"]
        self.pann_version = self.pann_versions[1]
        # CNN 10
        self.output_embedding: bool = False
    
        # CBAM
        self.use_cbam = True
        self.cbam_channels = 64
        self.cbam_reduction_factor = 4
        self.cbam_kernel_size = 7
    
        # ViT
        self.output_1d_embeddings = False
        self.patch_size=8
        self.dim=1024
        self.depth=6
        self.heads=16
        self.mlp_dim=2048
        self.vit_dropout=0.1
        self.vit_emb_dropout=0.

class YOHO_hparams(hparams):
    def __init__(self) -> None:
        super().__init__()
        # (NHWC) (-1, 129, 20, 32)
        self.use_cbam = False
        self.cbam_channels = 32
        self.cbam_reduction_factor = 2
        self.cbam_kernel_size = 3
    
        # Patchify
        self.use_patches = True
    
        # UFO Attention
        self.use_ufo = True
        self.ufo_d_k=512
        self.ufo_d_v=512
        self.ufo_h=8
    
        # ParNet Attention Usage
        self.use_pna = True
        self.pna_channels = 32
    
        # MobileViT Attention Usage
        self.use_mva = True
        self.mva_in_channel=128
        self.mva_dim=512
        self.mva_kernel_size=3
        self.mva_patch_size=5

        self.use_mish_activation = True


class CoAtNet_hparams(hparams):
    def __init__(self) -> None:
        super().__init__()
        # (NHWC) (-1, 129, 20, 32)
        self.use_cbam = True
        self.cbam_channels = 32
        self.cbam_reduction_factor = 2
        self.cbam_kernel_size = 3