import os

import pandas as pd

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AdamW, BertConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch
from accelerate import Accelerator

import wandb

from text.dataset import TextDataset
from text.train_functions import train_one_epoch


os.environ['WANDB_SILENT'] = 'true'

batch_size = 20
n_epochs = 20
max_len = 128
weight_decay = 0.01
init_lr = 1e-5
fold_number = 0

group_name = wandb.util.generate_id()
wandb.init(project='bert_base', group=group_name, job_type=str(fold_number))

wandb.config.batch_size = batch_size
wandb.config.n_epochs = n_epochs
wandb.config.max_len = max_len
wandb.config.weight_decay = weight_decay
wandb.config.init_lr = init_lr

df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
train_dataset = TextDataset(df, df[df['fold_group'] != 0], max_len=max_len)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataset = TextDataset(df, df[df['fold_group'] == 0], max_len=max_len)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
accelerator = Accelerator()

configuration = BertConfig(max_position_embeddings=max_len)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=df['label_group'].nunique())
model.to(device)
model.train()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=init_lr)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

best_loss = 0

for epoch in range(3):
    train_loss = train_one_epoch(model, train_dataloader, optimizer, accelerator, device)

    wandb.log({'train_loss': train_loss, 'epoch': epoch})

    if train_loss > best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(), 'best_bt.pth')

wandb.finish()
