from models.CoAtNet import CoAtNet
from typing import Any
import torch
from torch import nn
from utils.torch_utils import compute_conv_transpose_kernel_size, compute_conv_kernel_size
from utils.pl_utils import LM
from models.attention.CBAM import CBAMBlock
from config import add_EAP_to_path, CoAtNet_hparams
add_EAP_to_path()

hp = CoAtNet_hparams()


class VOICeCoAtNet(LM):
    """CoAtNet model adapted for VOICe dataset
    """

    def __init__(self,
                 num_classes: int = hp.num_classes,
                 input_height: int = hp.input_height, input_width: int = hp.input_width,
                 use_cbam: bool = hp.use_cbam, cbam_reduction_factor: int = hp.cbam_reduction_factor, cbam_kernel_size: int = hp.cbam_kernel_size,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        if self.input_height > self.input_width:
            self.make_input_square = nn.ConvTranspose2d(
                1, 1, (1, compute_conv_transpose_kernel_size(self.input_width, self.input_height)))
            self.square_dim = self.input_height
        elif self.input_width > self.input_height:
            self.make_input_square = nn.ConvTranspose2d(
                1, 1, (compute_conv_transpose_kernel_size(self.input_height, self.input_width), 1))
            self.square_dim = self.input_width
        else:
            self.make_input_square = nn.Identity()
            self.square_dim = self.input_width

        self.increase_channels_to_3 = nn.Conv2d(1, 3, (2, 2))

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam_reduction_factor: int = cbam_reduction_factor
            self.cbam_kernel_size: int = cbam_kernel_size
            self.cbam = CBAMBlock(channel=3,
                                  reduction=self.cbam_reduction_factor, kernel_size=self.cbam_kernel_size)

        self.cn = CoAtNet((self.square_dim-1, self.square_dim-1), 3, num_blocks=[
                          2, 2, 3, 5, 2], channels=[64, 96, 192, 384, 768], num_classes=3*hp.num_classes)
        self.increase_1d_channels = nn.Conv1d(1, 9, 1)
        # with torch.no_grad():
        #     # pann will then output a 2d image/tensor: (batch_size, 1, height, width)
        #     random_inp = torch.rand(
        #         (1, 1, *(max(self.input_height, self.input_width),)*2))
        #     output = self.cn(random_inp)
        #     self.cn_output_channels = output.size(1)
        #     self.cn_output_height_width = output.size(2)

        

        # self.head = nn.Sequential(
        #     nn.Conv2d(self.cn_output_channels, 1, (compute_conv_kernel_size(self.cn_output_height_width, hp.num_subwindows), compute_conv_kernel_size(
        #         self.cn_output_height_width, 3*self.num_classes)))
        # )

    def forward(self, input):
        x = input.float()
        x = self.make_input_square(x)
        x = self.increase_channels_to_3(x)
        if self.use_cbam:
            x = self.cbam(x)
        x = self.cn(x)
        x = torch.unsqueeze(x, 1)
        x = self.increase_1d_channels(x)
        
        # x = self.head(x)
        return x
