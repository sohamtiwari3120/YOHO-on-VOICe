from typing import Any
import torch
from torch import nn
from models.convnext import convnext_tiny
from config import hparams

hp = hparams()

class VOICeConvNeXt(nn.Module):
    """ConvNeXt Model with output linear layer
    """

    def __init__(self,
                 num_classes: int = hp.num_classes,
                 input_height: int = hp.input_height, input_width: int = hp.input_width,
                 *args: Any, **kwargs: Any) -> None:

        super(VOICeConvNeXt, self).__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.convert_channels_to_3 = nn.Conv2d(1, 3, (1, 1))
        self.convnext = convnext_tiny(False)
        # output shape
        self.head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 3*self.num_classes)
        )

    def forward(self, input):
        x = self.convert_channels_to_3(input.float())
        x = self.convnext(x)
        x = self.head(x) # -> (batch_size, 3*num_classes)
        x = torch.unsqueeze(x, dim=-2) # -> (batch_size, 1(=num_subwindows), 3*num_classes)
        return x