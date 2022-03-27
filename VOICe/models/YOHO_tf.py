from typing import List
import os
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from config import hparams
from utils.types import depthwise_layers_type
from utils.tf_utils import MonitorSedF1CallbackTf
from utils.data_utils import envs

hp = hparams()

class YohoTF:
    def __init__(self,
                 expt_folder: str,
                 env: str = hp.env,
                 depthwise_layers: depthwise_layers_type = hp.depthwise_layers,
                 num_classes: int = hp.num_classes,
                 input_height: int = hp.input_height,
                 input_width: int = hp.input_width,
                 l2_bias_reg_first_conv2d: float = hp.l2_bias_reg_first_conv2d,
                 l2_bias_reg_remaining_conv2d: float = hp.l2_bias_reg_remaining_conv2d,
                 l2_kernel_reg_first_conv2d: float = hp.l2_kernel_reg_first_conv2d,
                 l2_kernel_reg_remaining_conv2d: float = hp.l2_kernel_reg_remaining_conv2d,
                 batch_norm_eps: float = hp.batch_norm_eps,
                 spatial_dropout: float = hp.spatial_dropout) -> None:

        self.model_ckpt_folder_path = expt_folder
        if env not in envs:
            raise Exception('Invalid environment type.')
        self.env = env
        self.depthwise_layers = depthwise_layers
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width

        inputs = tf.keras.Input(
            shape=(input_height, input_width, 1), name="mel_input")
        X = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=2, padding='same', use_bias=False,
                                   activation=None, name="layer1/conv",
                                   kernel_regularizer=l2(l2_kernel_reg_first_conv2d), bias_regularizer=l2(l2_bias_reg_first_conv2d))(inputs)
        X = tf.keras.layers.BatchNormalization(
            center=True, scale=False, epsilon=batch_norm_eps, name="layer1/bn")(X)
        X = tf.keras.layers.ReLU(name="layer1/relu")(X)
        # X = tf.keras.layers.SpatialDropout2D(0.5)(X)
        for i in range(len(depthwise_layers)):
            X = tf.keras.layers.DepthwiseConv2D(kernel_size=depthwise_layers[i][0], strides=depthwise_layers[i][1], depth_multiplier=1, padding='same', use_bias=False,
                                                activation=None, name="layer" + str(i + 2)+"/depthwise_conv")(X)
            X = tf.keras.layers.BatchNormalization(
                center=True, scale=False, epsilon=batch_norm_eps, name="layer" + str(i + 2)+"/depthwise_conv/bn")(X)
            X = tf.keras.layers.ReLU(
                name="layer" + str(i + 2)+"/depthwise_conv/relu")(X)
            X = tf.keras.layers.Conv2D(filters=depthwise_layers[i][2], kernel_size=[1, 1], strides=1, padding='same', use_bias=False, activation=None,
                                       name="layer" +
                                       str(i + 2)+"/pointwise_conv",
                                       kernel_regularizer=l2(l2_kernel_reg_remaining_conv2d), bias_regularizer=l2(l2_bias_reg_remaining_conv2d))(X)
            X = tf.keras.layers.BatchNormalization(
                center=True, scale=False, epsilon=batch_norm_eps, name="layer" + str(i + 2)+"/pointwise_conv/bn")(X)
            X = tf.keras.layers.ReLU(
                name="layer" + str(i + 2)+"/pointwise_conv/relu")(X)

            X = tf.keras.layers.SpatialDropout2D(spatial_dropout)(X)

        _, _, sx, sy = X.shape
        X = tf.keras.layers.Reshape((-1, int(sx * sy)))(X)
        pred = tf.keras.layers.Conv1D(
            3*self.num_classes, kernel_size=1, activation="sigmoid")(X)

        self.model = tf.keras.Model(
            name=f"YohoTF", inputs=inputs,
            outputs=[pred])

    def summary(self):
        self.model.summary()

    def train_and_validate(self, train_data, validation_data, loss_function, learning_rate: float = hp.learning_rate, epochs: int = hp.epochs, fit_verbose: int = hp.fit_verbose, callbacks: List[tf.keras.callbacks.Callback] = None):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate), loss=loss_function)
        if callbacks is None:
            callbacks = [MonitorSedF1CallbackTf(self.env, self.model_ckpt_folder_path),
                         tf.keras.callbacks.EarlyStopping(monitor=hp.tf_monitor,
                                                          min_delta=hp.min_delta, patience=hp.tf_patience)]
        self.model.fit(x=train_data,
                       validation_data=validation_data,
                       epochs=epochs,
                       callbacks=callbacks,
                       verbose=fit_verbose)

    def load_from_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise Exception(f'{ckpt_path} does not exist/not found.')
        self.model.load_weights(ckpt_path)
