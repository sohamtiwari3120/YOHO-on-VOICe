import numpy as np
import torch
from config import num_classes


def compute_conv_output(input_dim: int, dilation: int = 1, kernel: int = 1, stride: int = 1) -> int:
    """Auxiliary function to help calculate the resulting dimension after performing the convolution operation.

    Args:
        input_dim (int): Value of the input dimension
        dilation (int, optional): kernel convolution dilation value. Defaults to 1.
        kernel (int, optional): kernel size. Defaults to 1.
        stride (int, optional): stride of convolution operation. Defaults to 1.

    Returns:
        int: resulting output dimension after performing conv operation
    """
    output_dim = np.floor((input_dim - dilation * (kernel - 1) - 1)/stride + 1)
    return output_dim


def loss_function(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Computes Squared Error for the true and predicted values. For each class, after computing the squared error, multiplies it by the probility of that sound event occuring (from y_true). Finally returns the aggregate loss.

    Args:
        y_true (torch.Tensor): true values
        y_pred (torch.Tensor): predicted values
    """
    # y.shape: (NUM_BATCHES, CHANNELS, 3*NUM_CLASSES) / (NUM_BATCHES, 9, 9)
    squared_difference = torch.square(y_true - y_pred)

    probability_multiplier = torch.ones_like(squared_difference)
    for i in range(num_classes):
        # multiply squared difference of start time for event i by the prob of event i occuring
        probability_multiplier[:, :, 3*i+1] = y_true[:, :, 3*i]
        # multiply squared difference of end time for event i by the prob of event i occuring
        probability_multiplier[:, :, 3*i+2] = y_true[:, :, 3*i]

    squared_difference = torch.multiply(
        squared_difference, probability_multiplier)  # element wise multiplication
    # Note the `axis=-1`
    return torch.sum(squared_difference, dim=[-1, -2])
