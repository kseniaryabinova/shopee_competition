import time

from torch import nn


def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion):
    total_loss = 0
    start_time = time.time()
    iter_counter = 0

    model.train()
    optimizer.zero_grad()

    for images, labels in dataloader:
        outputs = model(images.cuda(), labels.cuda())

        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        iter_counter += 1

    total_loss /= iter_counter

    return total_loss, time.time() - start_time
