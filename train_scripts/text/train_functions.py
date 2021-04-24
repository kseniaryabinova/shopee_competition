import random

import os

import numpy as np
import torch


def train_one_epoch(model, train_dataloader, optimizer, device):
    model.train()
    total_loss = 0
    iter_counter = 0

    for input_ids, attention_masks, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to(device),
                        attention_mask=attention_masks.to(device),
                        labels=labels.to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        iter_counter += 1

    total_loss /= iter_counter
    return total_loss


def train_one_epoch_acc(model, train_dataloader, optimizer, accelerator):
    model.train()
    total_loss = 0
    iter_counter = 0

    for input_ids, attention_masks, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_masks,
                        labels=labels)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        iter_counter += 1

    total_loss /= iter_counter
    return total_loss


def train_one_epoch_arc_bert(
        model,
        train_dataloader,
        criterion,
        optimizer,
        accelerator
):
    model.train()
    total_loss = 0
    iter_counter = 0

    for input_ids, attention_masks, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_masks,
                        labels=labels)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()

        total_loss += loss.item()
        iter_counter += 1

    total_loss /= iter_counter
    return total_loss


def seed_everything(seed=25):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
