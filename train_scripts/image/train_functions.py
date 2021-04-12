import time
import warnings

from sklearn.metrics import f1_score, pairwise_distances
from sklearn.neighbors import NearestNeighbors

import numpy as np
from numpy import dot
from numpy.linalg import norm

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import _LRScheduler


def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion, device,
                    scaler, scheduler=None, iters_to_accumulate=1,
                    clip_grads=False):
    total_loss = 0
    start_time = time.time()
    iter_counter = 0
    predictions = []
    targets = []
    softmax = nn.Softmax(dim=1)

    model.train()
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(dataloader):
        if scaler is not None:
            with autocast():
                outputs = model(images.to(device), labels.to(device))
                loss = criterion(outputs, labels.to(device))
                loss = loss / iters_to_accumulate
            scaler.scale(loss).backward()

            if (i + 1) % iters_to_accumulate == 0:
                if clip_grads:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), 1000)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        prediction = softmax(outputs).cpu().detach().numpy()
        prediction = np.argmax(prediction, axis=1)
        predictions.extend(prediction.tolist())
        targets.extend(labels.tolist())

        total_loss += loss.item()
        iter_counter += 1

    total_loss /= iter_counter / iters_to_accumulate
    f1 = get_metric(predictions, targets)

    return total_loss, time.time() - start_time, f1


def get_metric(predictions, targets):
    f1 = f1_score(targets, predictions, average='macro')
    return f1


def evaluate(model: nn.Module, dataloader, criterion, device):
    total_loss = 0
    start_time = time.time()
    iter_counter = 0
    predictions = []
    targets = []
    softmax = nn.Softmax(dim=1)

    model.eval()
    model.float()

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            outputs = model(images.to(device), labels.to(device))
            loss = criterion(outputs, labels.to(device))

            prediction = softmax(outputs).cpu().detach().numpy()
            prediction = np.argmax(prediction, axis=1)
            predictions.extend(prediction.tolist())
            targets.extend(labels.tolist())

            total_loss += loss.item()
            iter_counter += 1

    total_loss /= iter_counter
    f1 = get_metric(predictions, targets)

    return total_loss, time.time() - start_time, f1


def get_embeddings(model: nn.Module, dataloader, device):
    model.eval()
    model.float()
    embeddings = None

    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images.to(device))

            if embeddings is None:
                embeddings = outputs.cpu().detach().numpy()
            else:
                embeddings = np.concatenate([embeddings, outputs.cpu().detach().numpy()],
                                            axis=0)

    return embeddings


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def validate_embeddings_f1(embeddings, df):
    import pandas as pd
    pd.options.mode.chained_assignment = None

    def get_f1_metric(col):
        def f1score(row):
            n = len(np.intersect1d(row.target, row[col]))
            return 2 * n / (len(row.target) + len(row[col]))

        return f1score

    tmp = df.groupby('label_group').posting_id.agg('unique').to_dict()
    df['target'] = df.label_group.map(tmp)

    max_f1 = 0
    for thresh in np.arange(0.5, 0, -0.05):
        df['predictions'] = validate_with_knn(embeddings, df, thresh)
        df['f1'] = df.apply(get_f1_metric('predictions'), axis=1)
        f1_mean = df['f1'].mean()

        if max_f1 < f1_mean:
            max_f1 = f1_mean

    return max_f1


def validate_with_knn(embeddings, df, threshold):
    knn = NearestNeighbors(n_neighbors=50, metric='precomputed',
                           algorithm='brute', n_jobs=None)
    distance_matrix = pairwise_distances(embeddings, metric="cosine")
    knn.fit(distance_matrix)
    distances, indices = knn.kneighbors(distance_matrix)

    preds = []
    for k in range(embeddings.shape[0]):
        IDX = np.where(distances[k, ] < threshold)[0]
        IDS = indices[k, IDX]
        o = df.iloc[IDS].posting_id.values
        preds.append(o)

    return preds


class ShopeeScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_start=5e-6, lr_max=1e-5,
                 lr_min=1e-6, lr_ramp_ep=5, lr_sus_ep=0, lr_decay=0.8,
                 last_epoch=-1):
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        super(ShopeeScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            self.last_epoch += 1
            return [self.lr_start for _ in self.optimizer.param_groups]

        lr = self._compute_lr_from_epoch()
        self.last_epoch += 1

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch < self.lr_ramp_ep:
            lr = ((self.lr_max - self.lr_start) /
                  self.lr_ramp_ep * self.last_epoch +
                  self.lr_start)

        elif self.last_epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max

        else:
            lr = ((self.lr_max - self.lr_min) * self.lr_decay **
                  (self.last_epoch - self.lr_ramp_ep - self.lr_sus_ep) +
                  self.lr_min)
        return lr
