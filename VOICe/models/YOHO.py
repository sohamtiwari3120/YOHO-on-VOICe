from audioop import add
from matplotlib.style import use
import torch
from torch.nn import functional as F
from torch import nn
from typing import Any, List, Tuple
from utils.types import depthwise_layers_type
from models.attention.CBAM import CBAMBlock
from config import YOHO_hparams, add_EAP_to_path
from utils.torch_utils import compute_conv_output_dim, compute_padding_along_dim, InitializedBatchNorm2d, InitializedKerv2d, InitializedConv2d, InitializedConv1d
add_EAP_to_path()
from model.attention.UFOAttention import *
hp = YOHO_hparams()


class Yoho(nn.Module):
    """PyTorch Model for Yoho Algorithm
    """

    def __init__(self,
                 depthwise_layers: depthwise_layers_type = hp.depthwise_layers,
                 num_classes: int = hp.num_classes,
                 input_height: int = hp.input_height, input_width: int = hp.input_width,
                 use_cbam: bool = hp.use_cbam, cbam_channels: int = hp.cbam_channels, cbam_reduction_factor: int = hp.cbam_reduction_factor, cbam_kernel_size: int = hp.cbam_kernel_size,
                 use_patches: bool = hp.use_patches, use_ufo: bool = hp.use_ufo,
                 *args: Any, **kwargs: Any) -> None:

        super(Yoho, self).__init__(*args, **kwargs)
        self.depthwise_layers = depthwise_layers
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        output_width = self.input_width
        output_height = self.input_height

        self.use_patches = use_patches
        if self.use_patches:
            self.block_first = nn.Sequential(
                # making patches of input image
                InitializedConv2d(1, 32, (3, 3), stride=3, bias=False),
                InitializedBatchNorm2d(32, eps=1e-4),
                nn.ReLU()
            )
        else:
            self.block_first = nn.Sequential(
                InitializedConv2d(1, 32, (3, 3), stride=2, bias=False),
                InitializedBatchNorm2d(32, eps=1e-4),
                nn.ReLU()
            )

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam_channels: int = cbam_channels
            self.cbam_reduction_factor: int = cbam_reduction_factor
            self.cbam_kernel_size: int = cbam_kernel_size
            self.cbam = CBAMBlock(
                channel=self.cbam_channels, reduction=self.cbam_reduction_factor, kernel_size=self.cbam_kernel_size)

        padding_left_right = compute_padding_along_dim(
            self.input_width, 3, 2, 'same')
        padding_top_bottom = compute_padding_along_dim(
            self.input_height, 3, 2, 'same')
        # (padding_left, padding_right, padding_top, padding_bottom)
        self.block_first_padding: Tuple[int, int, int, int] = (
            padding_left_right[0], padding_left_right[1], padding_top_bottom[0], padding_top_bottom[1])

        output_width = compute_conv_output_dim(
            output_width, kernel=3, stride=3 if self.use_patches else 2, padding=padding_left_right)
        output_height = compute_conv_output_dim(
            output_height, kernel=3, stride=3 if self.use_patches else 2, padding=padding_top_bottom)

                
        self.use_ufo = use_ufo
        if self.use_ufo:
            self.ufo = UFOAttention(d_model=output_height, d_k=hp.ufo_d_k, d_v=hp.ufo_d_v, h=hp.ufo_h)

        self.blocks_depthwise = nn.ModuleList([])
        self.blocks_depthwise_padding: List[Tuple[int, int, int, int]] = []
        for i in range(len(self.depthwise_layers)):
            arr = self.depthwise_layers[i]
            if i > 0:
                prev = self.depthwise_layers[i-1]
            kernel_size = arr[0]
            stride = arr[1]
            output_channels = arr[2]
            if i == 0:
                input_channels = 32
            else:
                input_channels = prev[2]

            padding_left_right = compute_padding_along_dim(
                output_width, kernel_size[1], stride, 'same')
            padding_top_bottom = compute_padding_along_dim(
                output_height, kernel_size[0], stride, 'same')
            self.blocks_depthwise_padding.append(
                (padding_left_right[0], padding_left_right[1], padding_top_bottom[0], padding_top_bottom[1]))
            self.blocks_depthwise.append(
                nn.Sequential(
                    InitializedConv2d(input_channels, input_channels, kernel_size, stride=stride,
                                      padding=0, groups=input_channels, bias=False),  # step 1
                    InitializedBatchNorm2d(input_channels, eps=1e-4),
                    nn.ReLU(),
                    InitializedConv2d(input_channels, output_channels,
                                      (1, 1), 1, padding=0, bias=False),  # step 2
                    InitializedBatchNorm2d(output_channels, eps=1e-4),
                    nn.ReLU(),
                    nn.Dropout2d(0.1)
                )
            )
            # for step 1:
            output_width = compute_conv_output_dim(
                output_width, kernel=kernel_size[1], stride=stride, padding=padding_top_bottom)
            output_height = compute_conv_output_dim(
                output_height, kernel=kernel_size[0], stride=stride, padding=padding_top_bottom)

            # for step 2
            output_width = output_width  # since 1x1 conv with padding same and stride 1
            output_height = output_height  # since 1x1 conv with padding same and stride 1

        # (batch_size, num_channels, height, width)
        num_channels_last_depthwise = self.depthwise_layers[-1][-1]
        self.block_final = nn.Sequential(
            InitializedConv1d(int(output_width * num_channels_last_depthwise),
                              3*self.num_classes, 1)
        )
        if output_height != hp.num_subwindows:
            self.make_dim_num_sw = nn.Conv1d(int(output_height), hp.num_subwindows, 1)

    def forward(self, input):
        x = input.float()
        x = F.pad(x, self.block_first_padding)
        x = self.block_first(x)
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_ufo:
            x = self.ufo(x)
        for i, block in enumerate(self.blocks_depthwise):
            x = F.pad(x, self.blocks_depthwise_padding[i])
            x = block(x)
        batch_size, channels, height, width = x.size()
        x = torch.permute(x, (0, 1, 3, 2)).reshape(
            batch_size, channels*width, height)
        x = self.block_final(x)
        # Output: (N, C, L) = (N, 3*num_classes, num_subwindows)
        # Stored label output format: (N, L, C) = (N, num_subwindows, 3*num_classes)
        x = torch.permute(x, (0, 2, 1))
        if height != hp.num_subwindows:
            x = self.make_dim_num_sw(x)
        return x
