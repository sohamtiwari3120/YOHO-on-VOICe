from tkinter.tix import Tree
from typing import List, Optional, Tuple, Union
import os
import numpy as np
import torch
from config import num_classes, window_len_secs, num_classes, num_subwindows, rev_class_dict, batch_size, snr
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from utils.evaluate_utils import compute_sed_f1_errorrate
from utils.data_utils import file_paths, construct_audio_windows, convert_path_to_mono, get_log_melspectrogram, merge_sound_events


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
        for i in range(num_classes):
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


def convert_model_preds_to_soundevents(preds: np.ndarray, window_len_secs: float = window_len_secs, num_subwindows: int = num_subwindows, num_classes: int = num_classes, win_ranges: Optional[List[List[float]]] = None) -> List[Tuple[float, float, str]]:
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
                        [start, end, rev_class_dict[k]])

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


class MonitorSedF1Callback(Callback):
    """PyTorch Lightning Callback for monitoring f1 scores for sed task and storing model weights for best f1 scores and best error rates.

    Args:
        Callback (pytorch_lightning.callbacks.Callback): PyTorch Lightning Callback base class
    """

    def __init__(self, env):
        super(MonitorSedF1Callback, self).__init__()
        self.best_f1 = 0.0
        self.best_error = np.inf
        self.env = env

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        epoch = trainer.current_epoch
        reference_files = []
        estimated_files = []
        if epoch > 1:
            for audio_path in file_paths['validation'][self.env]:
                mono_audio_path = convert_path_to_mono(audio_path)

                unified_sound_events = predict_audio_path(
                    pl_module, mono_audio_path)
                folder_path = os.path.join(os.path.dirname(
                    audio_path), 'validation_predictions')
                os.makedirs(folder_path, exist_ok=True)

                reference_files.append(audio_path.replace('.wav', '.txt'))
                file_name = os.path.basename(audio_path).replace(
                    '.wav', "-se-prediction.txt")
                file_path = os.path.join(folder_path, file_name)
                estimated_files.append(file_path)

                with open(file_path, 'w') as fp:
                    fp.write('\n'.join('{},{},{}'.format(round(x[0], 5), round(
                        x[1], 5), x[2]) for x in unified_sound_events))

            curr_f1, curr_error = compute_sed_f1_errorrate(
                reference_files, estimated_files)
            self.log('f1', curr_f1)
            self.log('error_rate', curr_error)

            if curr_f1 > self.best_f1:
                self.best_f1 = curr_f1
                trainer.save_checkpoint(
                    f"./model_checkpoints/{snr}-mono/model-{self.env}-best-f1.ckpt")

            if curr_error < self.best_error:
                self.best_error = curr_error
                trainer.save_checkpoint(
                    f"./model_checkpoints/{snr}-mono/model-{self.env}-best-error.ckpt")

            print("F-measure: {:.3f} vs {:.3f}".format(curr_f1, self.best_f1))
            print("Error rate: {:.3f} vs {:.3f}".format(
                curr_error, self.best_error))

            # Or print all metrics as reports
