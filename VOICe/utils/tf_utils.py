from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
from utils.data_utils import envs, data_mode, file_paths, get_logmel_label_paths
from config import batch_size, input_height, input_width, num_subwindows, num_classes
import glob
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


class MyCustomCallback_44(tf.keras.callbacks.Callback):
    """
      callback when validating/testing
    """

    def __init__(self, env):
        super(MyCustomCallback_44, self).__init__()
        self.best_f1 = 0.0
        self.best_error = np.inf
        self.env = env

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        reference_files = []
        estimated_files = []
        if epoch > 1:
            for ii, audio in enumerate(eval(f"mono_{self.env}_validation_files")):
                audio_file_path = audio
                see = mk_preds_YOHO_mel(self.model, ii, self.env)
                n_label = n_label = f"./9dB-mono/eval-files-2/{self.env}/" + os.path.basename(
                    audio_file_path).replace(".wav", "") + "-se-prediction.txt"

                with open(n_label, 'w') as fp:
                    fp.write('\n'.join('{},{},{}'.format(
                        round(x[0], 5), round(x[1], 5), x[2]) for x in see))

            destination = f"./9dB-mono/eval-files-2/{self.env}/"
            test_set = glob.glob(destination + "*[0-9].txt")

            eval_path = "./9dB-mono/"

            file_list = [
                {
                    'reference_file': tt,
                    'estimated_file': tt.replace(".txt", "-se-prediction.txt")
                }
                for tt in test_set
            ]

            data = []

            # Get used event labels
            all_data = dcase_util.containers.MetaDataContainer()
            for file_pair in file_list:
                reference_event_list = sed_eval.io.load_event_list(
                    filename=file_pair['reference_file']
                )
                estimated_event_list = sed_eval.io.load_event_list(
                    filename=file_pair['estimated_file']
                )

                data.append({'reference_event_list': reference_event_list,
                            'estimated_event_list': estimated_event_list})

                all_data += reference_event_list

            event_labels = all_data.unique_event_labels

            # Start evaluating

            # Create metrics classes, define parameters
            segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=event_labels,
                time_resolution=1.0
            )

            event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=event_labels,
                t_collar=1.0
            )

            # Go through files
            for file_pair in data:
                segment_based_metrics.evaluate(
                    reference_event_list=file_pair['reference_event_list'],
                    estimated_event_list=file_pair['estimated_event_list']
                )

                event_based_metrics.evaluate(
                    reference_event_list=file_pair['reference_event_list'],
                    estimated_event_list=file_pair['estimated_event_list']
                )

            # Get only certain metrics
            overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
            curr_f1 = overall_segment_based_metrics['f_measure']['f_measure']
            curr_error = overall_segment_based_metrics['error_rate']['error_rate']

            if curr_f1 > self.best_f1:
                self.best_f1 = curr_f1
#         self.model.save_weights(f"./9dB-mono/model-{self.env}-best-f1.h5")

            if curr_error < self.best_error:
                self.best_error = curr_error
#         self.model.save_weights(f"./9dB-mono/model-{self.env}-best-error.h5")

            print("F-measure: {:.3f} vs {:.3f}".format(curr_f1, self.best_f1))
            print("Error rate: {:.3f} vs {:.3f}".format(
                curr_error, self.best_error))

            # Or print all metrics as reports


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
        X = np.array([np.load(self.logmel_npy[i])[:, None] for i in self.indices[idx *
                     self.batch_size: (idx+1)*self.batch_size]])  # to convert (H, W) -> (H, W, C)
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
