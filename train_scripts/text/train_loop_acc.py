import os

import pandas as pd
import wandb
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW, BertConfig

from text.dataset import TextDataset
from text.model import BERTWithArcFace
from text.train_functions import train_one_epoch_acc, seed_everything, \
    train_one_epoch_arc_bert, get_embeddings, validate_embeddings_f1

seed_everything(seed=25)

os.environ['WANDB_SILENT'] = 'true'

batch_size = 16
n_epochs = 20
max_len = 128
weight_decay = 0.01
init_lr = 1e-5
fold_number = 0

accelerator = Accelerator()

if accelerator.is_main_process:
    group_name = wandb.util.generate_id()
    wandb.init(project='bert_base', group=group_name, job_type=str(fold_number))
    wandb.config.model_name = 'bert-base-multilingual-cased'
    wandb.config.batch_size = batch_size
    wandb.config.n_epochs = n_epochs
    wandb.config.max_len = max_len
    wandb.config.weight_decay = weight_decay
    wandb.config.init_lr = init_lr

df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
train_dataset = TextDataset(df[df['fold_group'] != 0], max_len=max_len)
train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
valid_dataset = TextDataset(df[df['fold_group'] == 0], max_len=max_len)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

configuration = BertConfig(max_position_embeddings=max_len)
# model = BertForSequenceClassification.from_pretrained(
#     pretrained_model_name_or_path="bert-base-multilingual-cased",
#     num_labels=df[df['fold_group'] != 0]['label_group'].nunique()
# )
model = BERTWithArcFace(
    labels_num=df[df['fold_group'] != 0]['label_group'].nunique(),
    emb_size=512,
    dropout=0.0
)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=init_lr)
criterion = CrossEntropyLoss()

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader
)

best_loss = 0

for epoch in range(n_epochs):
    # train_loss = train_one_epoch_acc(model, train_dataloader, optimizer, accelerator)
    train_loss = train_one_epoch_arc_bert(
        model,
        train_dataloader,
        criterion,
        optimizer,
        accelerator
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        embeddings = get_embeddings(model, valid_dataloader, accelerator.device)
        f1, thresh = validate_embeddings_f1(embeddings, df[df['fold_group'] == 0])

        wandb.log({
            'train_loss': train_loss,
            'f1': f1,
            'thresh': thresh,
            'epoch': epoch
        })
        if train_loss > best_loss:
            best_loss = train_loss
            # accelerator.unwrap_model(model).save_pretrained('bt_vanilla')
            accelerator.save(accelerator.unwrap_model(model).state_dict(),
                             'best_bt_af_cased.pth')

if accelerator.is_main_process:
    wandb.finish()
