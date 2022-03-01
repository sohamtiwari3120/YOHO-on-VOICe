from numpy import double
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from typing import Any, List, Tuple
from utils.types import depthwise_layers_type
from utils.torch_utils import compute_conv_output_dim, loss_function, compute_padding_along_dim
from config import learning_rate, num_classes, input_height, input_width, depthwise_layers, mode, patience, factor


class YohoModel(LightningModule):
    """PyTorch (Lightning) model for YOHO algorithm

    Args:
        LightningModule (LightningModule): pytorch lightning class
    """

    def __init__(self,
                 depthwise_layers: depthwise_layers_type = depthwise_layers,
                 num_classes: int = num_classes,
                 input_height: int = input_height, input_width: int = input_width, learning_rate: double = learning_rate,
                 *args: Any, **kwargs: Any) -> None:

        super(YohoModel, self).__init__(*args, **kwargs)
        self.depthwise_layers = depthwise_layers
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.learning_rate = learning_rate
        output_width = self.input_width
        output_height = self.input_height
        self.block_first = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), stride=2, bias=False),
            nn.BatchNorm2d(32, eps=1e-4),
            nn.ReLU()
        )
        padding_left_right = compute_padding_along_dim(
            self.input_width, 3, 2, 'same')
        padding_top_bottom = compute_padding_along_dim(
            self.input_height, 3, 2, 'same')
        # (padding_left, padding_right, padding_top, padding_bottom)
        self.block_first_padding: Tuple[int, int, int, int] = (
            padding_left_right[0], padding_left_right[1], padding_top_bottom[0], padding_top_bottom[1])

        output_width = compute_conv_output_dim(
            output_width, kernel=3, stride=2, padding=padding_left_right)
        output_height = compute_conv_output_dim(
            output_height, kernel=3, stride=2, padding=padding_top_bottom)

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
                    nn.Conv2d(input_channels, input_channels, kernel_size, stride=stride,
                              padding='valid', groups=input_channels, bias=False),  # step 1
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
            nn.Conv1d(int(output_width * num_channels_last_depthwise),
                      3*self.num_classes, 1)
        )

    def forward(self, x):
        x = x.float()
        x = F.pad(x, self.block_first_padding)
        x = self.block_first(x)
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
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        sigmoid = torch.sigmoid(logits)
        loss = loss_function(y, sigmoid)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        sigmoid = torch.sigmoid(logits)
        loss = loss_function(y, sigmoid)
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(opt, mode=mode, patience=patience, factor=factor),
                "monitor": "validation_loss",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def predict(self, x):
        with torch.no_grad():
            # x.shape (n, c, h, w)
            x = torch.Tensor(x).to(self.device)
            logits = self(x).cpu()
            y = torch.sigmoid(logits)
            return y
