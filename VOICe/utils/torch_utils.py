from typing import Any, List, Optional, Tuple, Union
import os
import numpy as np
import torch
from torch import nn
from config import hparams
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from utils.evaluate_utils import compute_sed_f1_errorrate
from utils.data_utils import file_paths, construct_audio_windows, convert_path_to_mono, get_log_melspectrogram, merge_sound_events
from models.kervolution_pytorch import KernelConv2d, LinearKernel, PolynomialKernel, GaussianKernel

hp = hparams()


def compute_conv_transpose_kernel_size(input_dim: int, output_dim: int, stride: int = 1, padding: Union[int, Tuple[int, int]] = 0, output_padding: Union[int, Tuple[int, int]] = 0, dilation: int = 1) -> int:
    """Compute the kernel size for convolution transpose along a particular dim for obtaining the desired output_dim given the input_dim and other associated parameters.

    Args:
        input_dim (int): Length of the input along desired dim/axis
        output_dim (int): Desired length of the output along desired dim/axis
        stride (int, optional): Convolution Transpose kernel stride. Defaults to 1.
        padding (Union[int, Tuple[int, int]], optional): Convolution Transpose input padding. Defaults to 0.
        output_padding (Union[int, Tuple[int, int]], optional): Convolution Transpose output padding. Defaults to 0.
        dilation (int, optional): Convolution transpose dilation. Defaults to 1.

    Returns:
        int: kernel size along desired dim/axis
    """
    if isinstance(padding, int):
        padding = [padding] * 2
    if isinstance(output_padding, int):
        output_padding = [output_padding] * 2

    kernel_size = int((output_dim - ((input_dim-1)*stride-(padding[0]+padding[1])+(
        output_padding[0]+output_padding[1])+1))//dilation)+1
    return kernel_size

def compute_conv_kernel_size(input_dim: int, output_dim: int, stride: int = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: int = 1) -> int:
    """Compute the kernel size for convolution along a particular dim for obtaining the desired output_dim given the input_dim and other associated parameters.

    Args:
        input_dim (int): Length of the input along desired dim/axis
        output_dim (int): Desired length of the output along desired dim/axis
        stride (int, optional): Convolution kernel stride. Defaults to 1.
        padding (Union[int, Tuple[int, int]], optional): Convolution input padding. Defaults to 0.
        dilation (int, optional): Convolution dilation. Defaults to 1.

    Returns:
        int: kernel size along desired dim/axis
    """
    if isinstance(padding, int):
        padding = [padding] * 2

    kernel_size = (((output_dim - 1) * stride) + 1 - padding[0] - input_dim - padding[1])//(-dilation) + 1

    return int(kernel_size)

def compute_kernel_size_auto(input_dim: int, output_dim: int) -> int:
    """Given input and output lengths along a particular input dim, automatically selects whether to perform normal or transpose convolution and returns appropriate kernel size for the same.

    Args:
        input_dim (int): Length of the input along desired dim/axis
        output_dim (int): Desired length of the output along desired dim/axis

    Returns:
        int: kernel size along desired dim/axis for either normal or transposed convolution.
    """    
    if input_dim == output_dim:
        kernel_size = 1
    elif input_dim < output_dim:
        # need to perform transposed convolution
        kernel_size = compute_conv_transpose_kernel_size(input_dim=input_dim, output_dim=output_dim)
    else:
        # need to perform normal convolution
        kernel_size = compute_conv_kernel_size(input_dim=input_dim, output_dim=output_dim)
    return int(kernel_size)

def compute_conv_output_dim(input_dim: int, padding: Union[int, str, Tuple[int, int]] = 'valid', dilation: int = 1, kernel: int = 1, stride: int = 1) -> int:
    """Auxiliary function to help calculate the resulting dimension after performing the convolution operation.

    Args:
        input_dim (int): Value of the input dimension
        padding (Union[int, str, Tuple[int, int]], optional): Padding to be applied to input. Defaults to 'valid'.
        dilation (int, optional): kernel convolution dilation value. Defaults to 1.
        kernel (int, optional): kernel size. Defaults to 1.
        stride (int, optional): stride of convolution operation. Defaults to 1.

    Returns:
        int: resulting output dimension after performing conv operation
    """
    if isinstance(padding, str):
        if padding == 'valid':
            # i.e. no padding
            output_dim = np.floor(
                (input_dim - dilation * (kernel - 1) - 1)/stride + 1)
        elif padding == 'same':
            output_dim = np.ceil(input_dim/stride)
        else:
            raise Exception(
                f"Invalid padding={padding} value. Allowed string values are 'same', 'valid'")
    elif isinstance(padding, int):
        output_dim = np.floor(
            (input_dim + 2*padding - dilation*(kernel - 1) - 1)/stride + 1)
    elif isinstance(padding, tuple):
        if len(padding) == 2:
            output_dim = np.floor(
                (padding[0] + input_dim + padding[1] - dilation*(kernel - 1) - 1)/stride + 1)
        else:
            raise Exception(
                f'Invalid padding={padding} tuple. Only provide tuple of length 2.')
    else:
        raise Exception(
            f"Invalid padding={padding} value passed. Please provide either one of ['same', 'valid', <int>, (<int>, <int>)]")

    return output_dim


def compute_padding_along_dim(input_dim: int, kernel: int = 1, stride: int = 1, padding: str = 'same') -> Tuple[int, int]:
    """Compute the amount of padding before and after along a particular dimension for an input tensor, for different types of padding.

    Args:
        input_dim (int): Size of input dimension
        kernel (int, optional): Size of kernel along input_dimension. Defaults to 1.
        stride (int, optional): Value of stride along input_dimension. Defaults to 1.
        padding (str, optional): Different types of padding. One of 'valid' or 'same'. Defaults to 'same'.

    Raises:
        Exception: Invalid padding string value passed. Allowed string values are 'same', 'valid'.

    Returns:
        Tuple[int, int]: Tuple of padding values (padding_before, padding_after)
    """
    if padding == 'same':
        if (input_dim % stride == 0):
            pad_along_dim = max(kernel - stride, 0)
        else:
            pad_along_dim = max(kernel - (input_dim % stride), 0)
        pad_before = pad_along_dim // 2
        pad_after = pad_along_dim - pad_before
        return (int(pad_before), int(pad_after))
    elif padding == 'valid':
        return (0, 0)
    else:
        raise Exception(
            f"Invalid padding={padding} value. Allowed string values are 'same', 'valid'.")


def my_loss_fn(y_true, y_pred):
    squared_difference = torch.square(y_true - y_pred)
    ss_True = squared_difference[:, :, 0] * 0 + 1
    ss_0 = y_true[:, :, 0]  # (5, 5)
    ss_1 = y_true[:, :, 3]  # (5, 5)
    ss_2 = y_true[:, :, 6]  # (5, 5)
    sss = torch.stack((ss_True, ss_0, ss_0, ss_True, ss_1, ss_1,
                       ss_True, ss_2, ss_2,), dim=2)
    squared_difference = torch.multiply(
        squared_difference, sss)  # element wise multiplication
    # Note the `axis=-1`
    return torch.sum(squared_difference, dim=(-1, -2)).mean()


def mse(y_true: torch.Tensor, y_pred: torch.Tensor, weighted: bool = False) -> torch.Tensor:
    """Computes Mean Sum of Squared Error for the true and predicted values. For each class, after computing the squared error, multiplies it by the probility of that sound event occuring (from y_true). Finally returns the aggregate loss.

    Args:
        y_true (torch.Tensor): true values
        y_pred (torch.Tensor): predicted values
        weighted (bool): Whether to multiply squared difference by probability of sound event occuring or not.
    """
    # y.shape: (NUM_BATCHES, CHANNELS, 3*NUM_CLASSES) / (NUM_BATCHES, 9, 9)
    squared_difference = torch.square(y_true - y_pred)
    if weighted:
        probability_multiplier = torch.ones_like(squared_difference)
        for i in range(hp.num_classes):
            # multiply squared difference of start time for event i by the prob of event i occuring
            probability_multiplier[:, :, 3*i+1] = y_true[:, :, 3*i]
            # multiply squared difference of end time for event i by the prob of event i occuring
            probability_multiplier[:, :, 3*i+2] = y_true[:, :, 3*i]

        squared_difference = torch.multiply(
            squared_difference, probability_multiplier)  # element wise multiplication

    loss = torch.sum(squared_difference, dim=(-1, -2))
    return loss.mean()

# try changing loss to l1
# data input and output could be problematic
# print parameters of every layer, if not changing, then not training


def weighted_mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Computes Weighted Sum of Squared Error for the true and predicted values. For each class, after computing the squared error, multiplies it by the probility of that sound event occuring (from y_true). Finally returns the aggregate loss.

    Args:
        y_true (torch.Tensor): true values
        y_pred (torch.Tensor): predicted values
    """
    return mse(y_true, y_pred, True)


def convert_model_preds_to_soundevents(preds: np.ndarray, window_len_secs: float = hp.window_len_secs, num_subwindows: int = hp.num_subwindows, num_classes: int = hp.num_classes, win_ranges: Optional[List[List[float]]] = None) -> List[Tuple[float, float, str]]:
    """Converts model compatible annotations into human readable annotation, [event_start_time, event_end_time, event_name]. Note: 0<=event_start_time, event_end_time<=window_len_secs

    Args:
        preds (np.ndarray): Model compatible anns returned by model.predict() for a particular audio. Usually for a particular audio, there are multiple audio windows, consequently multiple predictions. NOTE: It should be detached from device.
        window_len_secs (float, optional): Length of audio window. Defaults to window_len_secs.
        num_subwindows (int, optional): Number of consecutive bins in compatible anns predicted by model. Defaults to num_subwindows.
        num_classes (int, optional): Number of classes in dataset. Defaults to num_classes.
        win_ranges (List[float, float], optional): List of window ranges for each window of the input audio file. Defaults to None.

    Returns:
        List[float, float, str]: List of human readable sound event, with event boundaries in [0, window_len_secs]
    """
    sound_events = []
    bin_length = window_len_secs / num_subwindows
    for i in range(len(preds)):
        # len(preds) = num_batches = number of input audio spectrograms to predict function.
        p = preds[i, :, :]
        events_curr_audio_win = []
        for bin in range(len(p)):
            for k in range(num_classes):
                if p[bin][k*3] >= 0.5:
                    start = bin_length * bin + bin_length * \
                        p[bin][k*3+1]
                    end = p[bin][k*3+2] * bin_length + start
                    if win_ranges is not None:
                        start += win_ranges[i][0]
                        end += win_ranges[i][0]
                    events_curr_audio_win.append(
                        [start, end, hp.rev_class_dict[k]])

        sound_events += events_curr_audio_win
    return sound_events


def predict_audio_path(model, audio_path: str, channels_last: bool = False):
    """Get model predictions for input audio given audio_path

    Args:
        model: Trained Model. Note: Ideally, the model env should be same as the env of the input audio.
        audio_path (str): path of the audio, ideally should be mono audio

    Returns:
        List[float, float, str]: List of merged sound events with less precise sound boundaries.
    """
    audio_wins, window_ranges = construct_audio_windows(audio_path)
    if channels_last:
        logmels = np.array([get_log_melspectrogram(audio_win).T
                            for audio_win in audio_wins])  # (N, H, W, C)
        logmels = np.expand_dims(logmels, axis=3)
    else:
        logmels = np.array([get_log_melspectrogram(audio_win).T[None, :]
                            for audio_win in audio_wins])  # (N, C, H, W)
    preds = model.predict(logmels)
    if not isinstance(preds, np.ndarray):
        preds = preds.cpu().numpy()
    sound_events = convert_model_preds_to_soundevents(
        preds, win_ranges=window_ranges)
    unified_sound_events = merge_sound_events(sound_events)
    return unified_sound_events


def generate_save_predictions(model, data_mode, env):
    reference_files = []
    estimated_files = []
    for audio_path in file_paths[data_mode][env]:
        mono_audio_path = convert_path_to_mono(audio_path)

        unified_sound_events = predict_audio_path(
            model, mono_audio_path)
        folder_path = os.path.join(os.path.dirname(
            audio_path), f'{data_mode}_predictions')
        os.makedirs(folder_path, exist_ok=True)

        reference_files.append(audio_path.replace('.wav', '.txt'))
        file_name = os.path.basename(audio_path).replace(
            '.wav', "-se-prediction.txt")
        file_path = os.path.join(folder_path, file_name)
        estimated_files.append(file_path)

        with open(file_path, 'w') as fp:
            fp.write('\n'.join('{},{},{}'.format(round(x[0], 5), round(
                x[1], 5), x[2]) for x in unified_sound_events))
    return reference_files, estimated_files


class MonitorSedF1Callback(Callback):
    """PyTorch Lightning Callback for monitoring f1 scores for sed task and storing model weights for best f1 scores and best error rates.

    Args:
        Callback (pytorch_lightning.callbacks.Callback): PyTorch Lightning Callback base class
    """

    def __init__(self, env: str, expt_folder: str, model_name: str = hp.model_name):
        super(MonitorSedF1Callback, self).__init__()
        self.best_f1 = 0.0
        self.best_error = np.inf
        self.env = env
        self.model_name = model_name
        self.model_ckpt_folder_path = expt_folder
        os.makedirs(self.model_ckpt_folder_path, exist_ok=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch

        if epoch > 1:
            reference_files, estimated_files = generate_save_predictions(
                pl_module, 'validation', self.env)
            curr_f1, curr_error = compute_sed_f1_errorrate(
                reference_files, estimated_files)
            self.log('f1', curr_f1)
            self.log('error_rate', curr_error)

            if curr_f1 > self.best_f1:
                self.best_f1 = curr_f1
                trainer.save_checkpoint(os.path.join(
                    self.model_ckpt_folder_path, f"model-{self.model_name}-{self.env}-best-f1.ckpt"))

            if curr_error < self.best_error:
                self.best_error = curr_error
                trainer.save_checkpoint(os.path.join(
                    self.model_ckpt_folder_path, f"model-{self.model_name}-{self.env}-best-error.ckpt"))

            print("F-measure: {:.3f} vs {:.3f}".format(curr_f1, self.best_f1))
            print("Error rate: {:.3f} vs {:.3f}".format(
                curr_error, self.best_error))

            # Or print all metrics as reports


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


class InitializedConv1d(nn.Conv1d):
    """Conv1d layer initalized using init_layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, initialize_layer=hp.initialize_layer) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)
        self.initialize_layer = initialize_layer
        if(self.initialize_layer):
            init_layer(self)


class InitializedKerv2d(KernelConv2d):
    """Kervolutional 2D layer initalized using init_layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=None, padding_mode='zeros', initialize_layer=hp.initialize_layer, kernel_fn=LinearKernel, *args: Any, **kwargs: Any):
        super().__init__(in_channels, out_channels, kernel_size, kernel_fn, stride,
                         padding, dilation, groups, bias, padding_mode, *args, **kwargs)
        self.initialize_layer = initialize_layer
        if(self.initialize_layer):
            init_layer(self)


class InitializedConv2d(nn.Conv2d):
    """Conv2d layer initalized using init_layer
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device=None, dtype=None, initialize_layer=True) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)
        self.initialize_layer = initialize_layer
        if(self.initialize_layer):
            init_layer(self)


class InitializedBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d layer initalized using init_bn
    """

    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, initialize_layer=hp.initialize_layer):
        super().__init__(num_features, eps, momentum,
                         affine, track_running_stats, device, dtype)
        self.initialize_layer = initialize_layer
        if(self.initialize_layer):
            init_bn(self)


# simply define a silu function
def serf(input):
    '''
    Applies the log-Softplus ERror activation Function (serf) element-wise:
        serf(x) = x * erf(ln(1+e^x))
    '''
    return input * torch.erf(torch.log(1+torch.exp(input))) # use torch.sigmoid to make sure that we created the most

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class Serf(nn.Module):
    '''
    Applies the log-Softplus ERror activation Function (serf) element-wise:
        serf(x) = x * erf(ln(1+e^x))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    References:
        -  Related paper:
        https://arxiv.org/pdf/2108.09598.pdf
    Examples:
        >>> m = serf()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__() # init the base class

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return serf(input) # simply apply already implemented Serf

class Residual(nn.Module):
    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def forward(self, x):
        return x + self.func(x)