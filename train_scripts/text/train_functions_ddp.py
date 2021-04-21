import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AdamW, BertConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch

import wandb

from text.dataset import TextDataset
from text.train_functions import train_one_epoch


def train_function(gpu, world_size, node_rank, gpus, fold_number, group_name):
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    torch.manual_seed(25)
    np.random.seed(25)

    rank = node_rank * gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    batch_size = 64
    n_epochs = 20
    max_len = 128
    weight_decay = 0.01
    init_lr = 1e-5

    if rank == 0:
        wandb.init(project='shopee_effnet0', group=group_name, job_type=str(fold_number))

        wandb.config.batch_size = batch_size
        wandb.config.n_epochs = n_epochs
        wandb.config.max_len = max_len
        wandb.config.weight_decay = weight_decay
        wandb.config.init_lr = init_lr

    df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
    train_dataset = TextDataset(df, df[df['fold_group'] != fold_number], max_len=max_len)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size // world_size, shuffle=False, num_workers=4,
                                  sampler=sampler)
    valid_dataset = TextDataset(df, df[df['fold_group'] == fold_number], max_len=max_len)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=df['label_group'].nunique())
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True, output_device=gpu)
    model.train()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=init_lr)

    best_loss = 0

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, device)

        if rank == 0:
            wandb.log({'train_loss': train_loss, 'epoch': epoch})

            if train_loss > best_loss:
                best_loss = train_loss
                torch.save(model.module.state_dict(), 'best_bt.pth')

    if rank == 0:
        wandb.finish()
