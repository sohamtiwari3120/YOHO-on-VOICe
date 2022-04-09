from typing import Any
import torch
from torch import nn
from utils.torch_utils import compute_conv_transpose_kernel_size, compute_conv_kernel_size
from config import add_EAP_to_path, hparams
add_EAP_to_path()
from model.attention.CoAtNet import CoAtNet

hp = hparams()


class VOICeCoAtNet(nn.Module):
    """CoAtNet model adapted for VOICe dataset
    """

    def __init__(self,
                 num_classes: int = hp.num_classes,
                 input_height: int = hp.input_height, input_width: int = hp.input_width,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        if self.input_height > self.input_width:
            self.make_input_square = nn.ConvTranspose2d(
                1, 1, (1, compute_conv_transpose_kernel_size(self.input_width, self.input_height)))
        else:
            self.make_input_square = nn.ConvTranspose2d(
                1, 1, (compute_conv_transpose_kernel_size(self.input_height, self.input_width), 1))

        self.cn = CoAtNet(in_ch=1, image_size=max(
            self.input_height, self.input_width))

        with torch.no_grad():
            # pann will then output a 2d image/tensor: (batch_size, 1, height, width)
            random_inp = torch.rand(
                (1, *(max(self.input_height, self.input_width),)*2))
            output = self.cn(random_inp)
            self.cn_output_channels = output.size(1)
            self.cn_output_height_width = output.size(2)

        self.head = nn.Sequential(
            nn.Conv2d(self.cn_output_channels, 1, (compute_conv_kernel_size(self.cn_output_height_width, hp.num_subwindows), compute_conv_kernel_size(
                self.cn_output_height_width, 3*self.num_classes)))
        )

    def forward(self, input):
        x = input.float()
        x = self.make_input_square(x)
        x = self.cn(x)
        x = self.head(x)
        x = torch.squeeze(x, 1)
        return x
