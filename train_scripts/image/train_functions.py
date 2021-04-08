import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from torch.cuda.amp import autocast
from torch.nn.utils import clip_grad_norm_


def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion, device, scaler, iters_to_accumulate=1,
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

