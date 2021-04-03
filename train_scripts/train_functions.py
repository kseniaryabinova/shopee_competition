import time

import numpy as np
from torch import nn
from sklearn.metrics import f1_score


def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion):
    total_loss = 0
    start_time = time.time()
    iter_counter = 0
    predictions = []
    targets = []
    softmax = nn.Softmax(dim=1)

    model.train()
    optimizer.zero_grad()

    for images, labels in dataloader:
        outputs = model(images.cuda(), labels.cuda())

        prediction = softmax(outputs).cpu().detach().numpy()
        prediction = np.argmax(prediction, axis=1)
        predictions.extend(prediction.tolist())
        targets.extend(labels.tolist())

        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        iter_counter += 1

    total_loss /= iter_counter
    f1 = get_metric(predictions, targets)

    return total_loss, time.time() - start_time, f1


def get_metric(predictions, targets):
    f1 = f1_score(targets, predictions, average='macro')
    return f1
