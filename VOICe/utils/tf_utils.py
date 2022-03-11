import tensorflow as tf
import os
import numpy as np
import glob

from utils.evaluate_utils import compute_sed_f1_errorrate
from config import num_classes, snr, shuffle, batch_size, input_height, input_width, num_subwindows
from utils.data_utils import convert_path_to_mono, file_paths, envs, data_mode, get_logmel_label_paths, sort_nicely
from utils.torch_utils import predict_audio_path
from utils.SpecAugment import spec_augment_tensorflow

def sse(y_true: tf.Tensor, y_pred: tf.Tensor, weighted: bool = False) -> tf.Tensor:
    """Computes Mean Sum of Squared Error for the true and predicted values. For each class, after computing the squared error, multiplies it by the probility of that sound event occuring (from y_true). Finally returns the aggregate loss.

    Args:
        y_true (tf.Tensor): true values
        y_pred (tf.Tensor): predicted values
        weighted (bool): Whether to multiply squared difference by probability of sound event occuring or not.
    """
    # y.shape: (NUM_BATCHES, CHANNELS, 3*NUM_CLASSES) / (NUM_BATCHES, 9, 9)
    squared_difference = tf.square(y_true - y_pred)
    if weighted:
        probability_multiplier = tf.ones_like(squared_difference)
        for i in range(num_classes):
            # multiply squared difference of start time for event i by the prob of event i occuring
            probability_multiplier[:, :, 3*i+1] = y_true[:, :, 3*i]
            # multiply squared difference of end time for event i by the prob of event i occuring
            probability_multiplier[:, :, 3*i+2] = y_true[:, :, 3*i]

        squared_difference = tf.multiply(
            squared_difference, probability_multiplier)  # element wise multiplication
        # Note the `axis=-1`
    return tf.reduce_sum(squared_difference, axis=[-1, -2])


def weighted_sse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes Weighted Sum of Squared Error for the true and predicted values. For each class, after computing the squared error, multiplies it by the probility of that sound event occuring (from y_true). Finally returns the aggregate loss.

    Args:
        y_true (tf.Tensor): true values
        y_pred (tf.Tensor): predicted values
    """
    return sse(y_true, y_pred, True)

def my_loss_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    ss_True = squared_difference[:, :, 0] * 0 + 1
    ss_0 = y_true[:, :, 0]  # (5, 5)
    ss_1 = y_true[:, :, 3]  # (5, 5)
    ss_2 = y_true[:, :, 6]  # (5, 5)
    sss = tf.stack((ss_True, ss_0, ss_0, ss_True, ss_1, ss_1,
                    ss_True, ss_2, ss_2,), axis=2)
    squared_difference = tf.multiply(
        squared_difference, sss)  # element wise multiplication
    # Note the `axis=-1`
    return tf.reduce_sum(squared_difference, axis=[-1, -2])

class MonitorSedF1CallbackTf(tf.keras.callbacks.Callback):
    """Tensorflow Callback for monitoring f1 scores for sed task and storing model weights for best f1 scores and best error rates.

    Args:
        Callback (tf.keras.callbacks.Callback): Tensorflow Callback base class
    """

    def __init__(self, env):
        super(MonitorSedF1CallbackTf, self).__init__()
        self.best_f1 = 0.0
        self.best_error = np.inf
        self.env = env
        self.model_ckpt_folder_path = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'model_checkpoints', f'{snr}-mono', 'tf')
        os.makedirs(self.model_ckpt_folder_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None) -> None:
        reference_files = []
        estimated_files = []
        
        if epoch > 1:
            for audio_path in file_paths['validation'][self.env]:
                mono_audio_path = convert_path_to_mono(audio_path)

                unified_sound_events = predict_audio_path(
                    self.model, mono_audio_path, channels_last=True)
                folder_path = os.path.join(os.path.dirname(
                    audio_path), 'tf_validation_predictions', self.env)
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

            if curr_f1 > self.best_f1:
                self.best_f1 = curr_f1
                self.model.save_weights(os.path.join(
                    self.model_ckpt_folder_path, f"model-{self.env}-best-f1.ckpt"))

            if curr_error < self.best_error:
                self.best_error = curr_error
                self.model.save_weights(os.path.join(
                    self.model_ckpt_folder_path, f"model-{self.env}-best-error.ckpt"))

            print("F-measure: {:.3f} vs {:.3f}".format(curr_f1, self.best_f1))
            print("Error rate: {:.3f} vs {:.3f}".format(
                curr_error, self.best_error))

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, mode: str, env: str, spec_transform=False, shuffle: bool = shuffle, batch_size: int = batch_size):
        """Initialises the VOICe dataset class to load data for given mode and env. (the logmel_path and label_path variables are env and mode specific.)

        Args:
            mode (str): One of ['training', 'test', 'validation']
            env (str): Either one of ['vehicle', 'outdoor', 'indoor']
            spec_transform (bool, optional): SpecAugmentation for spectrograms performed if true. Defaults to False.
            shuffle (bool, optional): If True, will shuffle indices at the end of every epoch. Defaults to shuffle.
            batch_size (int, optional): Number of samples in each batch. Defaults to batch_size.

        Raises:
        Exception: If invalid environment type chosen.
        Exception: If invalid data mode
        """
        if env not in envs:
            raise Exception('Invalid environment type.')
        if mode not in data_mode:
            raise Exception('Invalid data mode.')
        self.env = env
        self.mode = mode
        self.logmel_path, self.label_path = get_logmel_label_paths(
            self.mode, self.env)
        self.spec_transform = spec_transform

        self.logmel_npy = glob.glob(self.logmel_path+f'/logmelspec-*.npy')
        sort_nicely(self.logmel_npy)
        self.label_npy = glob.glob(self.label_path+f'/label-*.npy')
        sort_nicely(self.label_npy)
        self.indices = list(range(len(self.logmel_npy)))
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # print("The self.list_examples is {}".format(self.list_examples))
        return int(np.floor(len(self.logmel_npy) / self.batch_size))

    def __getitem__(self, idx):
        'Generate one batch of data'
        X = np.array([np.load(self.logmel_npy[i]) for i in self.indices[idx *
                     self.batch_size: (idx+1)*self.batch_size]])  # to convert (H, W) -> (H, W, C)
        # (height, width, channels=1)
        X = np.expand_dims(X, axis=3)
        assert X.shape == (self.batch_size, input_height, input_width, 1)
        y = np.array([np.load(self.label_npy[i]) for i in self.indices[idx *
                     self.batch_size:(idx+1)*self.batch_size]])
        assert y.shape == (self.batch_size, num_subwindows, 3*num_classes)

        if self.spec_transform and self.mode == 'training':
            tau = X.shape[1]
            v = X.shape[2]

            warped_frequency_spectrogram = spec_augment_tensorflow.frequency_masking(
                X, v=v,  frequency_masking_para=8, frequency_mask_num=1)
            warped_frequency_time_sepctrogram = spec_augment_tensorflow.time_masking(
                warped_frequency_spectrogram, tau=tau, time_masking_para=25, time_mask_num=2)
            X = warped_frequency_time_sepctrogram

        if isinstance(X, tf.Tensor):
            X = tf.cast(X, dtype=tf.float32)
        elif isinstance(X, np.ndarray):
            X = X.astype(float)
        return X, y

    def on_epoch_end(self):
        print(f'Called datagenerator on_epoch_end with self.shuffle={self.shuffle}')
        if self.shuffle == True:
            np.random.shuffle(self.indices)
