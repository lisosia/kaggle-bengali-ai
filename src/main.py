import gc
import os
from pathlib import Path
import random
import sys
import time

from tqdm.notebook import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.display import display, HTML

import torch

# --- plotly ---
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# --- models ---
from sklearn.model_selection import KFold
#from sklearn import preprocessing

# --- import local modules ---
import config
C = config.get_config("./config/001_seresnext_mixup.yml")
from dataset import *
from model import *

# --- setup ---
pd.set_option('max_columns', 50)


##############################################################
# Training
##############################################################
train_dataset, valid_dataset = get_trainval_dataset()

# --- Model ---
device = torch.device(C.device)

##############################################################
# ignite utility
##############################################################
import json
import numpy
import os
from logging import getLogger
from time import perf_counter

import pandas as pd
import torch

# from ignite.engine.engine import Engine, Events
# from ignite.metrics import Average
# from chainer_chemistry.utils import save_json

# def save_json(filepath, params):
#     with open(filepath, 'w') as f:
#         json.dump(params, f, indent=4)


# class DictOutputTransform:
#     def __init__(self, key, index=0):
#         self.key = key
#         self.index = index

#     def __call__(self, x):
#         if self.index >= 0:
#             x = x[self.index]
#         return x[self.key]


# def create_trainer(classifier, optimizer, device):
#     classifier.to(device)

#     def update_fn(engine, batch):
#         classifier.train()
#         optimizer.zero_grad()
#         # batch = [elem.to(device) for elem in batch]
#         x, y = [elem.to(device) for elem in batch]
#         loss, metrics, pred_y = classifier(x, y)
#         loss.backward()
#         optimizer.step()
#         return metrics, pred_y, y
#     trainer = Engine(update_fn)

#     for key in classifier.metrics_keys:
#         Average(output_transform=DictOutputTransform(key)).attach(trainer, key)
#     return trainer


# def create_evaluator(classifier, device):
#     classifier.to(device)

#     def update_fn(engine, batch):
#         classifier.eval()
#         with torch.no_grad():
#             # batch = [elem.to(device) for elem in batch]
#             x, y = [elem.to(device) for elem in batch]
#             _, metrics, pred_y = classifier(x, y)
#             return metrics, pred_y, y
#     evaluator = Engine(update_fn)

#     for key in classifier.metrics_keys:
#         Average(output_transform=DictOutputTransform(key)).attach(evaluator, key)
#     return evaluator


# class LogReport:
#     def __init__(self, evaluator=None, dirpath=None, logger=None):
#         self.evaluator = evaluator
#         self.dirpath = str(dirpath) if dirpath is not None else None
#         self.logger = logger or getLogger(__name__)

#         self.reported_dict = {}  # To handle additional parameter to monitor
#         self.history = []
#         self.start_time = perf_counter()

#     def report(self, key, value):
#         self.reported_dict[key] = value

#     def __call__(self, engine):
#         elapsed_time = perf_counter() - self.start_time
#         elem = {'epoch': engine.state.epoch,
#                 'iteration': engine.state.iteration}
#         elem.update({f'train/{key}': value
#                      for key, value in engine.state.metrics.items()})
#         if self.evaluator is not None:
#             elem.update({f'valid/{key}': value
#                          for key, value in self.evaluator.state.metrics.items()})
#         elem.update(self.reported_dict)
#         elem['elapsed_time'] = elapsed_time
#         self.history.append(elem)
#         if self.dirpath:
#             save_json(os.path.join(self.dirpath, 'log.json'), self.history)
#             self.get_dataframe().to_csv(os.path.join(self.dirpath, 'log.csv'), index=False)

#         # --- print ---
#         msg = ''
#         for key, value in elem.items():
#             if key in ['iteration']:
#                 # skip printing some parameters...
#                 continue
#             elif isinstance(value, int):
#                 msg += f'{key} {value: >6d} '
#             else:
#                 msg += f'{key} {value: 8f} '
# #         self.logger.warning(msg)
#         print(msg)

#         # --- Reset ---
#         self.reported_dict = {}

#     def get_dataframe(self):
#         df = pd.DataFrame(self.history)
#         return df


# class SpeedCheckHandler:
#     def __init__(self, iteration_interval=10, logger=None):
#         self.iteration_interval = iteration_interval
#         self.logger = logger or getLogger(__name__)
#         self.prev_time = perf_counter()

#     def __call__(self, engine: Engine):
#         if engine.state.iteration % self.iteration_interval == 0:
#             cur_time = perf_counter()
#             spd = self.iteration_interval / (cur_time - self.prev_time)
#             self.logger.warning(f'{spd} iter/sec')
#             # reset
#             self.prev_time = cur_time

#     def attach(self, engine: Engine):
#         engine.add_event_handler(Events.ITERATION_COMPLETED, self)


# class ModelSnapshotHandler:
#     def __init__(self, model, filepath='model_{count:06}.pt',
#                  interval=1, logger=None):
#         self.model = model
#         self.filepath: str = str(filepath)
#         self.interval = interval
#         self.logger = logger or getLogger(__name__)
#         self.count = 0

#     def __call__(self, engine: Engine):
#         self.count += 1
#         if self.count % self.interval == 0:
#             filepath = self.filepath.format(count=self.count)
#             torch.save(self.model.state_dict(), filepath)
#             # self.logger.warning(f'save model to {filepath}...')

# import warnings
# import torch
# from ignite.metrics.metric import Metric

# class EpochMetric(Metric):
#     """Class for metrics that should be computed on the entire output history of a model.
#     Model's output and targets are restricted to be of shape `(batch_size, n_classes)`. Output
#     datatype should be `float32`. Target datatype should be `long`.

#     .. warning::

#         Current implementation stores all input data (output and target) in as tensors before computing a metric.
#         This can potentially lead to a memory error if the input data is larger than available RAM.


#     - `update` must receive output of the form `(y_pred, y)`.

#     If target shape is `(batch_size, n_classes)` and `n_classes > 1` than it should be binary: e.g. `[[0, 1, 0, 1], ]`.

#     Args:
#         compute_fn (callable): a callable with the signature (`torch.tensor`, `torch.tensor`) takes as the input
#             `predictions` and `targets` and returns a scalar.
#         output_transform (callable, optional): a callable that is used to transform the
#             :class:`~ignite.engine.Engine`'s `process_function`'s output into the
#             form expected by the metric. This can be useful if, for example, you have a multi-output model and
#             you want to compute the metric with respect to one of the outputs.

#     """

#     def __init__(self, compute_fn, output_transform=lambda x: x):

#         if not callable(compute_fn):
#             raise TypeError("Argument compute_fn should be callable.")

#         super(EpochMetric, self).__init__(output_transform=output_transform)
#         self.compute_fn = compute_fn

#     def reset(self):
#         self._predictions = torch.tensor([], dtype=torch.float32)
#         self._targets = torch.tensor([], dtype=torch.long)

#     def update(self, output):
#         y_pred, y = output
#         self._predictions = torch.cat([self._predictions, y_pred], dim=0)
#         self._targets = torch.cat([self._targets, y], dim=0)

#         # Check once the signature and execution of compute_fn
#         if self._predictions.shape == y_pred.shape:
#             try:
#                 self.compute_fn(self._predictions, self._targets)
#             except Exception as e:
#                 warnings.warn("Probably, there can be a problem with `compute_fn`:\n {}.".format(e),
#                               RuntimeWarning)

#     def compute(self):
#         return self.compute_fn(self._predictions, self._targets)


import numpy as np
import sklearn.metrics
import torch

def macro_recall(pred_y, y, n_grapheme=168, n_vowel=11, n_consonant=7):
    pred_y = torch.split(pred_y, [n_grapheme, n_vowel, n_consonant], dim=1)
    pred_labels = [torch.argmax(py, dim=1).cpu().numpy() for py in pred_y]

    y = y.cpu().numpy()
    # pred_y = [p.cpu().numpy() for p in pred_y]

    recall_grapheme = sklearn.metrics.recall_score(pred_labels[0], y[:, 0], average='macro')
    recall_vowel = sklearn.metrics.recall_score(pred_labels[1], y[:, 1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(pred_labels[2], y[:, 2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    # print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '
    #       f'total {final_score}, y {y.shape}')
    return final_score


def calc_macro_recall(solution, submission):
    # solution df, submission df
    scores = []
    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
        y_true_subset = solution[solution[component] == component]['target'].values
        y_pred_subset = submission[submission[component] == component]['target'].values
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score


#####################################################################
# --- Training setting ---
#####################################################################
import argparse
from distutils.util import strtobool
import os

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from numpy.random.mtrand import RandomState
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

class BengaliModule(pl.LightningModule):

    def __init__(self):
        super(BengaliModule, self).__init__()

        n_grapheme = 168
        n_vowel = 11
        n_consonant = 7
        n_total = n_grapheme + n_vowel + n_consonant
        print('n_total', n_total)
        # Set pretrained='imagenet' to download imagenet pretrained model...
        predictor = PretrainedCNN(in_channels=1, out_dim=n_total, model_name=C.model_name, pretrained=None)
        print('predictor', type(predictor))

        self.classifier = BengaliClassifier(predictor).to(device)

    def forward(self, x):
        return self.classifier(x)  # todo return [logi1, logi2, logi3]

    @staticmethod
    def accuracy(y, t):
        pred_label = torch.argmax(y, dim=1)
        count = pred_label.shape[0]
        correct = (pred_label == t).sum().type(torch.float32)
        acc = correct / count
        return acc.item(), pred_label.item()
        # return acc

    @staticmethod
    def _calc_loss_metric(preds, y0, y1, y2):
        """return loss(torch.Tensor) and log(not Tensor)"""
        # _loss_func = mixup_cross_entropy_loss if do_mixup else torch.nn.functional.cross_entropy
        loss_grapheme = mixup_cross_entropy_loss(preds[0], y0)
        loss_vowel = mixup_cross_entropy_loss(preds[1], y1)
        loss_consonant = mixup_cross_entropy_loss(preds[2], y2)
        loss = loss_grapheme + loss_vowel + loss_consonant

        acc_grapheme, y_hat0 = accuracy(preds[0], y0)
        acc_vowel, y_hat1 = accuracy(preds[1], y1)
        acc_consonant, y_hat2 = accuracy(preds[2], y2)
        logs = {
            'loss': loss.item(),
            'loss_grapheme': loss_grapheme.item(),
            'loss_vowel': loss_vowel.item(),
            'loss_consonant': loss_consonant.item(),
            'acc_grapheme': acc_grapheme,
            'acc_vowel': acc_vowel,
            'acc_consonant': acc_consonant,
        }
        return loss, logs, [y_hat0, y_hat1, y_hat2]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
        do_mixup = np.random.rand() > 0.5
        if do_mixup:
            x, y0, y1, y2 = mixup_multi_targets(x.cpu(), y0.cpu(), y1.cpu(), y2.cpu())
            x  = x.cuda()
            y0, y1, y2 = y0.cuda(), y1.cuda(), y2.cuda()

        preds = self.forward(x)
        loss, logs, _ = self._calc_loss_metric(preds, y0, y1, y2)

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

        preds = self.forward(x)

        _, logs, y_hat_arr = self._calc_loss_metric(preds, y0, y1, y2)

        return {'log' : logs, 
                'y_true' : y.cpu().numpy().values,
                'y_hat' : y_hat_arr}

    def validation_end(self, outputs):
        # OPTIONAL
        keys = outputs[0]['log'].keys()
        tf_logs = {}
        for key in keys:
            tf_logs['avg_' + key] = torch.stack([x['log'][key] for x in outputs]).mean()
        tf_logs['lr'] = self.scheduler.optimizer.param_groups[0]['lr']

        return {'avg_val_loss': tf_logs['avg_loss'], 'log': tf_logs}
        
    # def test_step(self, batch, batch_idx):
    #     # OPTIONAL
    #     x, y = batch
    #     y_hat = self.forward(x)
    #     return {'test_loss': F.cross_entropy(y_hat, y)}

    # def test_end(self, outputs):
    #     # OPTIONAL
    #     avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'test_loss': avg_loss}
    #     return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer =  torch.optim.Adam(self.classifier.parameters(), lr=0.001 * C.batch_size / 32)  # 0.001 for bs=32
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-10)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.num_workers)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(valid_dataset, batch_size=C.batch_size, shuffle=False, num_workers=C.num_workers)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)


m = BengaliModule()
trainer = Trainer()
trainer.fit(m)


### train_loader = DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.num_workers)
### valid_loader = DataLoader(valid_dataset, batch_size=C.batch_size, shuffle=False, num_workers=C.num_workers)

### optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001 * C.batch_size / 32)  # 0.001 for bs=32
### scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
###     optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-10)

# trainer = create_trainer(classifier, optimizer, device)
# def output_transform(output):
#     metric, pred_y, y = output
#     return pred_y.cpu(), y.cpu()
# EpochMetric(
#     compute_fn=macro_recall,
#     output_transform=output_transform
# ).attach(trainer, 'recall')

# pbar = ProgressBar()
# pbar.attach(trainer, metric_names='all')

# evaluator = create_evaluator(classifier, device)
# EpochMetric(
#     compute_fn=macro_recall,
#     output_transform=output_transform
# ).attach(evaluator, 'recall')

# def run_evaluator(engine):
#     evaluator.run(valid_loader)

# def schedule_lr(engine):
#     # metrics = evaluator.state.metrics
#     metrics = engine.state.metrics
#     avg_mae = metrics['loss']

#     # --- update lr ---
#     lr = scheduler.optimizer.param_groups[0]['lr']
#     scheduler.step(avg_mae)
#     log_report.report('lr', lr)

# trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
# trainer.add_event_handler(Events.EPOCH_COMPLETED, schedule_lr)
# log_report = LogReport(evaluator, C.outdir)
# trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)
# trainer.add_event_handler(
#     Events.EPOCH_COMPLETED,
#     ModelSnapshotHandler(predictor, filepath=C.outdir + '/predictor.pt'))

# #################################################################
# # Train
# #################################################################
# trainer.run(train_loader, max_epochs=C.n_epoch)

# train_history = log_report.get_dataframe()
# train_history.to_csv(C.outdir + '/log.csv', index=False)

# train_history
