import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_cmx(y_true, y_pred, outpath, figsize=(20,20)):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=figsize)
    sns.heatmap(df_cmx, annot=True)
    plt.savefig(outpath)


# https://github.com/PyTorchLightning/PyTorch-Lightning/blob/0.6.0/pytorch_lightning/callbacks/pt_callbacks.py#L165-L353
import os
import shutil
import logging
import warnings
import numpy as np

import pytorch_lightning.callbacks.pt_callbacks
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

class ModelCheckpointCustom(pytorch_lightning.callbacks.pt_callbacks.Callback):
    r"""
    Save the model after every epoch.
    Args:
        filepath (str): path to save the model file.
            Can contain named formatting options to be auto-filled.
            Example::
                # save epoch and val_loss in name
                ModelCheckpoint(filepath='{epoch:02d}-{val_loss:.2f}.hdf5')
                # saves file like: /path/epoch_2-val_loss_0.2.hdf5
        monitor (str): quantity to monitor.
        verbose (bool): verbosity mode, 0 or 1.
        save_top_k (int): if `save_top_k == k`,
            the best k models according to
            the quantity monitored will be saved.
            if `save_top_k == 0`, no models are saved.
            if `save_top_k == -1`, all models are saved.
            Please note that the monitors are checked every `period` epochs.
            if `save_top_k >= 2` and the callback is called multiple
            times inside an epoch, the name of the saved file will be
            appended with a version count starting with `v0`.
        mode (str): one of {auto, min, max}.
            If `save_top_k != 0`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only (bool): if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period (int): Interval (number of epochs) between checkpoints.
    Example::
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import ModelCheckpoint
        checkpoint_callback = ModelCheckpoint(filepath='my_path')
        Trainer(checkpoint_callback=checkpoint_callback)
        # saves checkpoints to my_path whenever 'val_loss' has a new min
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_top_k=1, save_weights_only=False,
                 mode='auto', period=1, prefix=''):
        super(ModelCheckpointCustom, self).__init__()
        if (
            save_top_k and
            os.path.isdir(filepath) and
            len(os.listdir(filepath)) > 0
        ):
            warnings.warn(
                f"Checkpoint directory {filepath} exists and is not empty with save_top_k != 0."
                "All files in this directory will be deleted when a checkpoint is saved!"
            )

        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        os.makedirs(filepath, exist_ok=True)
        self.save_top_k = save_top_k
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_check = 0
        self.prefix = prefix
        self.best_k_models = {}
        # {filename: monitor}
        self.kth_best_model = ''
        self.best = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(
                f'ModelCheckpoint mode {mode} is unknown, '
                'fallback to auto mode.', RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.kth_value = np.Inf
            self.mode = 'min'
        elif mode == 'max':
            self.monitor_op = np.greater
            self.kth_value = -np.Inf
            self.mode = 'max'
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.kth_value = -np.Inf
                self.mode = 'max'
            else:
                self.monitor_op = np.less
                self.kth_value = np.Inf
                self.mode = 'min'

    def _del_model(self, filepath):
        dirpath = os.path.dirname(filepath)

        # make paths
        os.makedirs(dirpath, exist_ok=True)

        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

    def _save_model(self, filepath):
        dirpath = os.path.dirname(filepath)

        # make paths
        os.makedirs(dirpath, exist_ok=True)

        # delegate the saving to the model
        self.save_function(filepath)

    def check_monitor_top_k(self, current):
        less_than_k_models = len(self.best_k_models.keys()) < self.save_top_k
        if less_than_k_models:
            return True
        return self.monitor_op(current, self.best_k_models[self.kth_best_model])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_check += 1

        if self.save_top_k == 0:
            # no models are saved
            return
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}.ckpt'
            version_cnt = 0
            while os.path.isfile(filepath):
                # this epoch called before
                filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}_v{version_cnt}.ckpt'
                version_cnt += 1

            if self.save_top_k != -1:
                current = logs.get(self.monitor)

                if current is None:
                    warnings.warn(
                        f'Can save best model only with {self.monitor} available,'
                        ' skipping.', RuntimeWarning)
                else:
                    if self.check_monitor_top_k(current):

                        # remove kth
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            delpath = self.kth_best_model
                            self.best_k_models.pop(self.kth_best_model)
                            self._del_model(delpath)

                        self.best_k_models[filepath] = current
                        if len(self.best_k_models.keys()) == self.save_top_k:
                            # monitor dict has reached k elements
                            if self.mode == 'min':
                                self.kth_best_model = max(self.best_k_models, key=self.best_k_models.get)
                            else:
                                self.kth_best_model = min(self.best_k_models, key=self.best_k_models.get)
                            self.kth_value = self.best_k_models[self.kth_best_model]

                        if self.mode == 'min':
                            self.best = min(self.best_k_models.values())
                        else:
                            self.best = max(self.best_k_models.values())
                        if self.verbose > 0:
                            logging.info(
                                f'\nEpoch {epoch:05d}: {self.monitor} reached'
                                f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                                f' {filepath} as top {self.save_top_k}')
                        self._save_model(filepath)

                    else:
                        if self.verbose > 0:
                            logging.info(
                                f'\nEpoch {epoch:05d}: {self.monitor}'
                                f' was not in top {self.save_top_k}')

            # else:
            if 1:
                filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch}_latest.ckpt'
                prev_filepath = f'{self.filepath}/{self.prefix}_ckpt_epoch_{epoch-1}_latest.ckpt'
                if self.verbose > 0:
                    logging.info(f'\nEpoch {epoch:05d}: saving model to {filepath}')
                self._del_model(prev_filepath)
                self._save_model(filepath)
