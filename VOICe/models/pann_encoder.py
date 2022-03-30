'''https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py'''

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from utils.torch_utils import compute_conv_transpose_kernel_size, compute_conv_kernel_size
from config import hparams
import os
from models.attention.CBAM import CBAMBlock

__author__ = "Soham Tiwari"
__credits__ = ["qiuqiangkong"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

hp = hparams()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn10(nn.Module):
    def __init__(self, output_embedding: bool = hp.output_embedding):

        super(Cnn10, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.output_embedding: bool = output_embedding
        if self.output_embedding:
            self.fc1 = nn.Linear(512, 512, bias=True)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        if self.output_embedding:
            init_layer(self.fc1)

    def forward(self, input):
        # 1. Try pooling/linear layer
        # 2. Or change bn0 to 128, but ideally avoid this step right
        # 3. Remove all intermediate layers between input and pann

        x = input  # -> (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)   # -> (batch_size, mel_bins, time_steps, 1)
        x = self.bn0(x)         # -> (batch_size, mel_bins, time_steps, 1)
        x = x.transpose(1, 3)   # -> (batch_size, 1, time_steps, mel_bins)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # (batch_size, 512, T/16, mel_bins/16)
        x = F.dropout(x, p=0.2, training=self.training)
        if self.output_embedding:
            x = torch.mean(x, dim=3)

            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            # embedding = F.dropout(x, p=0.5, training=self.training)
        return x


class Cnn14(nn.Module):
    def __init__(self):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)

    def forward(self, input, mixup_lambda=None):

        x = input.unsqueeze(1)   # (batch_size, 1, time_steps, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        return x


class Tag(nn.Module):
    def __init__(self, class_num):
        super(Tag, self).__init__()
        self.feature = Cnn10()
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, class_num, bias=True)
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc)

    def forward(self, input):
        '''
        :param input: (batch_size,time_steps, mel_bins)
        :return: ()
        '''
        x = self.feature(input)  # (batch_size, 512, T/16, mel_bins/16)
        x = torch.mean(x, dim=3)  # (batch_size, 512, T/16)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        # (batch_size,class_num)
        output = torch.sigmoid(self.fc(x))
        # output = self.fc(x)

        return output


class VOICePANN(nn.Module):
    """ConvNeXt Model with output linear layer
    """

    def __init__(self,
                 num_classes: int = hp.num_classes,
                 input_height: int = hp.input_height, input_width: int = hp.input_width,
                 pann_encoder_ckpt_path: str = hp.pann_encoder_ckpt_path, use_cbam: bool = hp.use_cbam, cbam_channels: int = hp.cbam_channels, cbam_reduction_factor: int = hp.cbam_reduction_factor, cbam_kernel_size: int = hp.cbam_kernel_size,
                 pann_output_embedding: bool = hp.output_embedding,
                 *args: Any, **kwargs: Any) -> None:

        super(VOICePANN, self).__init__(*args, **kwargs)
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAMBlock(
                channel=cbam_channels, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.pann_encoder_ckpt_path = pann_encoder_ckpt_path
        self.change_channels_to_64 = nn.Conv2d(self.input_width, 64, 1)
        self.pann_output_embedding = pann_output_embedding
        self.pann = Cnn10(self.pann_output_embedding)

        with torch.no_grad():
            if not self.pann_output_embedding:
                # pann will then output a 2d image/tensor: (batch_size, 1, height, width)
                random_inp = torch.rand(
                    (1, 1, self.input_height, 64))
                output = self.pann(random_inp)
                self.pann_output_height = output.size(2)
                self.pann_output_width = output.size(3)

        if os.path.exists(self.pann_encoder_ckpt_path):
            self.pann.load_state_dict(torch.load(self.pann_encoder_ckpt_path)[
                                      'model'], strict=False)
            print(
                f'loaded pann_cnn10 pretrained encoder state from {self.pann_encoder_ckpt_path}')
                
        if self.pann_output_embedding:
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 3*self.num_classes)
            )
        else:
            kernel = (1, compute_conv_transpose_kernel_size(self.pann_output_width, 3*self.num_classes))

            self.head = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel),
                nn.Conv2d(in_channels=256,
                          out_channels=128,
                          kernel_size=(compute_conv_kernel_size(self.pann_output_height, hp.num_subwindows), 1)),
                nn.Conv2d(in_channels=128,
                          out_channels=64,
                          kernel_size=(1, 1)),
                nn.Dropout(p = 0.2),
                nn.Conv2d(in_channels=64,
                          out_channels=3,
                          kernel_size=(1, 1)),
                nn.Dropout(p = 0.2),
                nn.Conv2d(in_channels=3,
                          out_channels=1,
                          kernel_size=(1, 1)
                          )
            )

    def forward(self, input):
        x = input.float()  # -> (batch_size, 1, num_frames, n_mels)
        x = x.transpose(1, 3)  # -> (batch_size, n_mels, num_frames, 1)
        x = self.change_channels_to_64(x)  # -> (batch_size, 64, num_frames, 1)
        if self.use_cbam:
            x = self.cbam(x)  # -> (batch_size, 64, num_frames, 1)
        x = x.transpose(1, 3)  # -> (batch_size, 1, num_frames, 64)
        x = self.pann(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.head(x)  # -> (batch_size, 3*num_classes)
        
        if self.pann_output_embedding:
            x = torch.unsqueeze(x, dim=-2) # -> (batch_size, 1(=num_subwindows), 3*num_classes)
        else:
           # -> (batch_size, 1, num_subwindows, 3*num_classes)
            x = torch.squeeze(x) # -> (batch_size, num_subwindows, 3*num_classes)
        return x
