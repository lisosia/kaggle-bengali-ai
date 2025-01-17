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

from tqdm import tqdm
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import sklearn.metrics
from sklearn.model_selection import KFold

import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

# --- import local modules ---
import config
if True:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train")
    test_parser = subparsers.add_parser("test")
    
    train_parser.add_argument('--config', required=True)
    args = parser.parse_args()
    C = config.get_config(args.config)

from dataset import *
from model import *

# --- setup ---
pd.set_option('max_columns', 50)


def macro_recall(y_true, pred_y, n_grapheme=168, n_vowel=11, n_consonant=7):
    recall_grapheme  = sklearn.metrics.recall_score(y_true[0], pred_y[0], average='macro')
    recall_vowel     = sklearn.metrics.recall_score(y_true[1], pred_y[1], average='macro')
    recall_consonant = sklearn.metrics.recall_score(y_true[2], pred_y[2], average='macro')
    scores = [recall_grapheme, recall_vowel, recall_consonant]
    final_score = np.average(scores, weights=[2, 1, 1])
    # print(f'recall: grapheme {recall_grapheme}, vowel {recall_vowel}, consonant {recall_consonant}, '
    #       f'total {final_score}, y {y.shape}')
    return {'recall/weight_mean': final_score,
            'recall/recall_grapheme': recall_grapheme,
            'recall/recall_vowel': recall_vowel,
            'recall/recall_consonant': recall_consonant}


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

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

LR_ARR = []
LOSS_ARR = []
def lr_range_test():
    INIT_LR = 1e-5
    WD = 1.25e-4
    # FACTOR = 10
    # FACTOR = np.sqrt(10)
    # FACTOR = np.sqrt(np.sqrt(10))
    FACTOR = 3.4
    predictor = PretrainedCNN(in_channels=1, out_dim=186, model_name=C.model_name, pretrained=None).to(C.device)
    classifier = BengaliClassifier(predictor).to(C.device)
    classifier.train()

    # optimizer =  torch.optim.Adam(classifier.parameters(), lr=INIT_LR, weight_decay=WD)
    optimizer =  torch.optim.AdamW(classifier.parameters(), lr=INIT_LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=FACTOR)

    # load data after model is in cuda
    train_dataset, valid_dataset = get_trainval_dataset_png()
    loader = DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.num_workers)

    i = 0
    for batch in loader:
        lr_now = scheduler.get_lr()
        LR_ARR.append(INIT_LR * FACTOR**i)
        
        x, y = batch
        x, y = x.to(C.device), y.to(C.device)
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

        ### do_cutmix  = np.random.rand() < 0.5
        do_cutmix  = False
        if do_cutmix:
            # x, y0, y1, y2 = mixup_multi_targets(x, y0, y1, y2)
            x, y0, y1, y2 = cutmix_multi_targets(x, y0, y1, y2, alpha=1.)
        else:
            y0, y1, y2 = onehot(y0, 168), onehot(y1, 11), onehot(y2, 7)

        optimizer.zero_grad()
        
        preds = classifier(x)
        loss_grapheme = mixup_cross_entropy_loss(preds[0], y0, class_dx=0)
        loss_vowel = mixup_cross_entropy_loss(preds[1], y1, class_dx=1)
        loss_consonant = mixup_cross_entropy_loss(preds[2], y2, class_dx=2)
        loss = loss_grapheme + loss_vowel + loss_consonant
        
        loss.backward()  # need this?
        optimizer.step()
        scheduler.step()
        
        LOSS_ARR.append(loss.item())

        i += 1
        if loss > 13 or i > 50 : break
lr_range_test()
print(LR_ARR)
print(LOSS_ARR)
plt.ylim()
plt.grid(True)
plt.plot(LR_ARR, LOSS_ARR, 'o--')
plt.xscale('log')
plt.savefig("lr_range_test.jpg")
exit()
        
#####################################################################
# --- Training Class ---
#####################################################################
class BengaliModule(pl.LightningModule):

    def __init__(self, hparams):
        super(BengaliModule, self).__init__()
        self.hparams = hparams
        self.device = torch.device(C.device)

        n_grapheme = 168
        n_vowel = 11
        n_consonant = 7
        n_total = n_grapheme + n_vowel + n_consonant
        print('n_total', n_total)

        # Set pretrained='imagenet' to download imagenet pretrained model...
        predictor = PretrainedCNN(in_channels=1, out_dim=n_total, model_name=C.model_name, pretrained=None).to(self.device)
        print('predictor', type(predictor))
        self.classifier = BengaliClassifier(predictor).to(self.device)

        # load data after model is in cuda
        self.train_dataset, self.valid_dataset = get_trainval_dataset_png()

    def forward(self, x):
        return self.classifier(x.to(self.device))  # todo return [logi1, logi2, logi3]

    @staticmethod
    def _calc_loss_metric(preds, y0, y1, y2, log_prefix):
        """return loss(torch.Tensor) and log(not Tensor)"""
        # _loss_func = mixup_cross_entropy_loss if do_mixup else torch.nn.functional.cross_entropy
        loss_grapheme = mixup_cross_entropy_loss(preds[0], y0, class_dx=0)
        loss_vowel = mixup_cross_entropy_loss(preds[1], y1, class_dx=1)
        loss_consonant = mixup_cross_entropy_loss(preds[2], y2, class_dx=2)
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
        x, y = x.to(self.device), y.to(self.device)
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

        do_cutmix  = np.random.rand() < 0.5
        if do_cutmix:
            # x, y0, y1, y2 = mixup_multi_targets(x, y0, y1, y2)
            x, y0, y1, y2 = cutmix_multi_targets(x, y0, y1, y2, alpha=1.)
        else:
            y0, y1, y2 = onehot(y0, 168), onehot(y1, 11), onehot(y2, 7)

        preds = self.forward(x)
        loss, logs, _ = self._calc_loss_metric(preds, y0, y1, y2, log_prefix='train')

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]
        y0, y1, y2 = onehot(y0, 168), onehot(y1, 11), onehot(y2, 7)

        preds = self.forward(x)

        _, logs, y_hat_arr = self._calc_loss_metric(preds, y0, y1, y2, log_prefix='val')

        return {'_val_log' : logs, 
                'y_true' : np.swapaxes(y.cpu().numpy(), 0, 1),  # shape is class(3), B
                'y_hat' : y_hat_arr
               }

    def validation_end(self, outputs):
        # OPTIONAL
        keys = outputs[0]['_val_log'].keys()
        tf_logs = {}
        for key in keys:
            tf_logs[key] = np.stack([x['_val_log'][key] for x in outputs]).mean()
        ### print( len ( self.trainer.lr_schedulers ))
        ### tf_logs['lr'] = self.trainer.lr_schedulers[0].optimizer.param_groups[0]['lr']

        y_true = [np.concatenate([x['y_true'][i] for x in outputs]) for i in range(3)]
        y_hat  = [np.concatenate([x['y_hat'][i]  for x in outputs]) for i in range(3)]
        recalls_dict = macro_recall(y_true, y_hat)
        tf_logs = {**tf_logs, **recalls_dict}  # merge dicts
        return {'val_loss': tf_logs['loss/val_total_loss'], 'log': tf_logs, 'progress_bar': tf_logs}
        
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
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-10, verbose=True)
        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.num_workers)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.valid_dataset, batch_size=C.batch_size, shuffle=False, num_workers=C.num_workers)

    # @pl.data_loader
    # def test_dataloader(self):
    #     # OPTIONAL
    #     return DataLoader(MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()), batch_size=32)

    def predict_proba(self, loader):
        arr0, arr1, arr2 = [], [], []
        for x in loader:
            with torch.no_grad():
                preds0, preds1, preds2 = self.forward(x.to(self.device))
            arr0.append(np.apply_along_axis(softmax, -1, preds0.cpu().numpy()))
            arr1.append(np.apply_along_axis(softmax, -1, preds1.cpu().numpy()))
            arr2.append(np.apply_along_axis(softmax, -1, preds2.cpu().numpy()))
        return np.concatenate(arr0), np.concatenate(arr1), np.concatenate(arr2)


def train(args):
    m = BengaliModule(args)
    trainer = pl.Trainer(
        early_stop_callback=None, max_epochs=C.n_epoch,
        fast_dev_run=False)
    trainer.fit(m)


# resutre trainer
    # trainer = pl.Trainer()
    # trainer.restore(path, torch.cuda.is_available())

# proper way to load is calling BengaliModule.load_from_checkpoint(path)
#   m = BengaliModule.load_from_checkpoint(path)
# It seems load_from_checkpoint() assume I passed hparams arg when training?
def _load(checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    m = BengaliModule()
    m.load_state_dict(checkpoint['state_dict'])
    return m

def test():
    path = './lightning_logs/version_5/checkpoints/_ckpt_epoch_27.ckpt'
    model = _load(path).to(C.device)
    model.eval()
    
    ret = model.forward(torch.Tensor(np.random.randn(3, 1, 128, 128)).to(C.device))
    print(ret[0].shape, ret[1].shape, ret[2].shape)
    
    preds0, preds1, preds2 = [], [], []
    for i in range(4):  # 4 parque loop
        indices = [i]
        test_images = prepare_image(
            C.datadir, C.featherdir, data_type='test', submission=True, indices=indices)
        n_dataset = len(test_images)
        print(f'i={i}, n_dataset={n_dataset}')
        test_dataset = BengaliAIDataset(
            test_images, None, 
            transform=Transform(affine=False, crop=True, size=(C.image_size, C.image_size), train=False)
        )

        # INFER
        ########################## TODO, CHANGE BACTHSIZE
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        p0arr, p1arr, p2arr = model.predict_proba(test_loader)
        # print("p[012]arr", p0arr, p1arr, p2arr)
        preds0.append(p0arr)
        preds1.append(p1arr)
        preds2.append(p2arr)

        # CLEANUP
        del test_loader
        del test_dataset
        del test_images
        gc.collect()

    preds0 = np.concatenate(preds0).argmax(axis=-1)
    preds1 = np.concatenate(preds1).argmax(axis=-1)
    preds2 = np.concatenate(preds2).argmax(axis=-1)
    print('shapes:', 'p0', preds0.shape, 'p1', preds1.shape, 'p2', preds2.shape)

    # SUBMISSION
    row_id = []
    target = []
    for i in tqdm(range(len(preds0))):
        row_id += [f'Test_{i}_grapheme_root', f'Test_{i}_vowel_diacritic',
               f'Test_{i}_consonant_diacritic']
        target += [preds0[i], preds1[i], preds2[i]]
    submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
    submission_df.to_csv('submission.csv', index=False)



if __name__ == "__main__":

    if args.subcommand == 'train':
        train(args)
    elif args.subcommand == 'test':
        test()
