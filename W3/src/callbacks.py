from typing import List
from typing import Union
import os

import tensorflow as tf
from keras.callbacks import CSVLogger


def get_callbacks(model_path: str, experiment_path: str, es_use: bool = True, es_patience: int = 15) -> List[tf.keras.callbacks.Callback]:
    """This function sets and returns a list of callbacks.

    :return: List of callbacks
    """
    callbacks = [
        get_model_checkpoint_callback(model_path),
        get_early_stopping(es_use, es_patience),
        CSVLogger(experiment_path + '/log.csv', append=True, separator=';')
    ]
    return callbacks


def get_model_checkpoint_callback(log_dir: str) -> tf.keras.callbacks.ModelCheckpoint:
    """This function sets a ModelCheckpoint Keras Callback.

    :param log_dir: String specifying the path to save the model
    :return: ModelCheckpoint Callback
    """
    return tf.keras.callbacks.ModelCheckpoint(
        log_dir,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )


def get_early_stopping(use: bool = True, patience: int = 15) -> Union[tf.keras.callbacks.EarlyStopping, None]:
    """This function sets a lr schedule and returns an Early Stopping Callback.

    :param use: Boolean specifying if Early Stopping should be used
    :param patience: Integer specifying the patience for Early Stopping
    :return:  Callback or None if not used
    """
    ret = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=patience,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        )
        if use
        else None
    )
    return ret
