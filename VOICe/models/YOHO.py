import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from typing import Any, List
from utils.types import depthwise_layers_type
from utils.torch_utils import compute_conv_output, loss_function
from config import learning_rate, num_classes, input_height, input_width, depthwise_layers

class YohoModel(LightningModule):
    """PyTorch (Lightning) model for YOHO algorithm

    Args:
        LightningModule ([type]): pytorch lightning class
    """

    def __init__(self,
                 depthwise_layers: depthwise_layers_type = depthwise_layers,
                 num_classes: int = num_classes,
                 input_height: int = input_height, input_width: int = input_width,
                 *args: Any, **kwargs: Any) -> None:

        super(YohoModel, self).__init__(*args, **kwargs)
        self.depthwise_layers = depthwise_layers
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        output_width = self.input_width

        self.block_first = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), 2, bias=False),
            nn.BatchNorm2d(32, eps=1e-4),
            nn.ReLU()
        )
        output_width = compute_conv_output(output_width, kernel=3, stride=2)

        self.blocks_depthwise: List[nn.Module] = []
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

            self.blocks_depthwise.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, input_channels, kernel_size, stride,
                              'same', groups=input_channels, bias=False),  # step 1
                    nn.BatchNorm2d(input_channels, eps=1e-4),
                    nn.ReLU(),
                    nn.Conv2d(input_channels, output_channels,
                              (1, 1), 1, 'same', bias=False),  # step 2
                    nn.BatchNorm2d(output_channels, eps=1e-4),
                    nn.ReLU(),
                    nn.Dropout2d(0.1)
                )
            )
            # for step 1:
            if stride == 1:
                output_width = output_width
            else:
                output_width = compute_conv_output(
                    output_width, kernel=kernel_size, stride=stride)

            # for step 2
            output_width = output_width  # since 1x1 conv with padding same and stride 1

        # (batch_size, num_channels, height, width)
        num_channels_last_depthwise = self.depthwise_layers[-1][-1]
        self.block_final = nn.Sequential(
            nn.Conv1d(output_width * num_channels_last_depthwise,
                      3*self.num_classes, 1)
        )

    def forward(self, x):
        x = self.block_first(x)
        x = self.blocks_depthwise(x)
        batch_size, channels, height, width = x.size()
        x = torch.permute(x, (0, 1, 3, 2)).view(
            batch_size, channels*width, height)
        x = self.block_final(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        sigmoid = F.sigmoid(logits)
        loss = loss_function(y, sigmoid)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=learning_rate)

    def predict(self, x):
        with torch.no_grad():
            # x.shape (n, c, h, w)
            logits = self(x).cpu()
            y = F.sigmoid(logits)
            return y