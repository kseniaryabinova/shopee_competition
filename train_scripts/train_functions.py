import time

from torch import nn


def train_one_epoch(model: nn.Module, dataloader, optimizer, criterion):
    total_loss = 0
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    for images, labels in dataloader:
        outputs = model(images.cuda(), labels.cuda())

        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    total_loss /= len(list(iter(dataloader)))

    return total_loss, time.time() - start_time
