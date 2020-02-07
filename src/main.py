import argparse
import gc
import os
from pathlib import Path
import random
import sys
import json
import numpy
import time
from time import perf_counter

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
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

import sklearn.metrics
from sklearn.model_selection import KFold

from distutils.util import strtobool
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

# --- import local modules ---
import config
C = config.get_config("./config/001_seresnext_mixup.yml")
from dataset import *
from model import *

# --- setup ---
pd.set_option('max_columns', 50)

train_dataset, valid_dataset = get_trainval_dataset()
device = torch.device(C.device)

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


def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    t = torch.argmax(t, dim=1)

    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc.item(), pred_label.tolist()
    # return acc


#####################################################################
# --- Training Class ---
#####################################################################
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
        return self.classifier(x.to(device))  # todo return [logi1, logi2, logi3]

    @staticmethod
    def _calc_loss_metric(preds, y0, y1, y2, log_prefix):
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
            f'loss/{log_prefix}_total_loss': loss.item(),
            f'loss/{log_prefix}_loss_grapheme': loss_grapheme.item(),
            f'loss/{log_prefix}_loss_vowel': loss_vowel.item(),
            f'loss/{log_prefix}_loss_consonant': loss_consonant.item(),
            f'acc/{log_prefix}_acc_grapheme': acc_grapheme,
            f'acc/{log_prefix}_acc_vowel': acc_vowel,
            f'acc/{log_prefix}_acc_consonant': acc_consonant,
        }
        return loss, logs, [y_hat0, y_hat1, y_hat2]

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        x, y = x.to(C.device), y.to(C.device)
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

        do_mixup = np.random.rand() > 0.5
        if do_mixup:
            x, y0, y1, y2 = mixup_multi_targets(x, y0, y1, y2)
        else:
            y0, y1, y2 = onehot(y0, 168), onehot(y1, 11), onehot(y2, 7)

        preds = self.forward(x)
        loss, logs, _ = self._calc_loss_metric(preds, y0, y1, y2, log_prefix='train')

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        x, y = x.to(C.device), y.to(C.device)
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
        y0, y1, y2 = onehot(y0, 168), onehot(y1, 11), onehot(y2, 7)

        preds = self.forward(x)

        _, logs, y_hat_arr = self._calc_loss_metric(preds, y0, y1, y2, log_prefix='val')

        return {'_val_log' : logs, 
                'y_true' : y.cpu().numpy(),
                'y_hat' : y_hat_arr}

    def validation_end(self, outputs):
        # OPTIONAL
        keys = outputs[0]['_val_log'].keys()
        tf_logs = {}
        for key in keys:
            tf_logs[key] = np.stack([x['_val_log'][key] for x in outputs]).mean()
            # tf_logs['avg_' + key] = np.stack([x['_val_log'][key] for x in outputs]).mean()
        ### print( len ( self.trainer.lr_schedulers ))
        ### tf_logs['lr'] = self.trainer.lr_schedulers[0].optimizer.param_groups[0]['lr']

        return {'val_loss': tf_logs['loss/val_total_loss'], 'log': tf_logs}
        
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

    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)


m = BengaliModule()
trainer = pl.Trainer()
trainer.fit(m)
