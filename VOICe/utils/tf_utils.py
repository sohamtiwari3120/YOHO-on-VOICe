from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from utils.data_utils import envs, data_mode, file_paths, get_logmel_label_paths
from config import batch_size, input_height, input_width, num_subwindows, num_classes
import glob
from utils.SpecAugment import spec_augment_tensorflow


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, mode: str, env: str, spec_transform=False, shuffle=shuffle, batch_size=batch_size):
        """Initialises the VOICe dataset class to load data for given mode and env. (the logmel_path and label_path variables are env and mode specific.)

        Args:
            mode (str): One of ['training', 'test', 'validation']
            env (str): Either one of ['vehicle', 'outdoor', 'indoor']
            spec_transform (bool, optional): SpecAugmentation for spectrograms performed if true. Defaults to False.

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
        self.label_npy = glob.glob(self.label_path+f'/label-*.npy')
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
        X = np.array([np.load(self.logmel_npy[i])[:, None] for i in self.indices[idx * self.batch_size : (idx+1)*self.batch_size]])  # to convert (H, W) -> (H, W, C)
        # (height, width, channels=1)
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
        if self.shuffle == True:
            np.random.shuffle(self.indices)
