from numpy import double
import torch
from torch.optim import Adam
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.core.lightning import LightningModule
from typing import Any
from config import hparams
from utils.torch_utils import weighted_mse, mse, my_loss_fn

hp = hparams()

class LM(LightningModule):
    """PyTorch (Lightning) Module for YOHO algorithm

    Args:
        LightningModule (LightningModule): pytorch lightning class
    """

    def __init__(self,
                 model: nn.Module,
                 learning_rate: double = hp.learning_rate, loss_function=eval(hp.loss_function_str),
                 *args: Any, **kwargs: Any) -> None:

        super(LM, self).__init__(*args, **kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = loss_function

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # -> (batch_size, num_windows, 3*num_classes)
        sigmoid = torch.sigmoid(logits)
        loss = self.loss_function(y, sigmoid)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        sigmoid = torch.sigmoid(logits)
        loss = self.loss_function(y, sigmoid)
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    # check for default parameter values for tf and pytorch
    def configure_optimizers(self):
        opt = Adam(self.model.parameters(),
                   lr=self.learning_rate, eps=hp.adam_eps, weight_decay=hp.adam_weight_decay)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(opt, mode=hp.mode, patience=hp.patience, factor=hp.factor),
                "monitor": hp.monitor,
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }

    # def on_fit_start(self) -> None:
    #     print(f"\n\n\n\nModel Params:\n")
    #     for param in self.model.parameters():
    #         print(param)
    #     print(f"\nEnd\n\n")

    # def on_fit_end(self) -> None:
    #     print(f"\n\n\n\nModel Params:\n")
    #     for param in self.model.parameters():
    #         print(param)
    #     print(f"\nEnd\n\n")

    def predict(self, x):
        with torch.no_grad():
            # x.shape (n, c, h, w)
            x = torch.Tensor(x).to(self.device)
            logits = self(x)
            y = torch.sigmoid(logits)
            return y
