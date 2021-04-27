import random
import os

import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

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

    for input_ids, attention_masks, labels, _ in train_dataloader:
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


def get_embeddings(model, dataloader, accelerator):
    model.eval()
    model.float()
    embeddings = None
    indices_array = None

    with torch.no_grad():
        for input_ids, attention_masks, _, indices in dataloader:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
            )

            if embeddings is None:
                embeddings = accelerator.gather(outputs).cpu().detach().numpy()
                indices_array = accelerator.gather(indices).cpu().detach().numpy()
            else:
                embs = accelerator.gather(outputs).cpu().detach().numpy()
                inds = accelerator.gather(indices).cpu().detach().numpy()
                embeddings = np.concatenate([embeddings, embs], axis=0)
                indices_array = np.concatenate([indices_array, inds], axis=0)
                print(embeddings.shape, indices_array.shape)

    return embeddings[indices_array]


def get_embeddings_with_device(model, dataloader, device):
    model.eval()
    model.float()
    embeddings = None

    with torch.no_grad():
        for input_ids, attention_masks, _, indices in dataloader:
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_masks.to(device),
            )

            if embeddings is None:
                embeddings = outputs.cpu().detach().numpy()
            else:
                embeddings = np.concatenate([
                    embeddings,
                    outputs.cpu().detach().numpy()
                ], axis=0)

    return embeddings


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
    best_thresh = 0
    for thresh in np.arange(0.5, 0, -0.05):
        df['predictions'] = validate_with_knn(embeddings, df, thresh)
        df['f1'] = df.apply(get_f1_metric('predictions'), axis=1)
        f1_mean = df['f1'].mean()

        if max_f1 < f1_mean:
            max_f1 = f1_mean
            best_thresh = thresh

    return max_f1, best_thresh


def validate_with_knn(embeddings, df, threshold):
    knn = NearestNeighbors(n_neighbors=50, metric='precomputed',
                           algorithm='brute', n_jobs=None)
    distance_matrix = pairwise_distances(embeddings, metric="cosine")
    knn.fit(distance_matrix)
    distances, indices = knn.kneighbors(distance_matrix)

    print(embeddings.shape, df.shape, distances.shape, indices.shape)

    preds = []
    for k in range(embeddings.shape[0]):
        IDX = np.where(distances[k, ] < threshold)[0]
        IDS = indices[k, IDX]
        o = df.iloc[IDS].posting_id.values
        preds.append(o)

    return preds
