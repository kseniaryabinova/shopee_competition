import pandas as pd

from transformers import BertForSequenceClassification, Trainer, TrainingArguments, AdamW, BertConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch

from text.dataset import TextDataset

df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
train_dataset = TextDataset(df, df[df['fold_group'] != 0])
train_dataloader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_dataset = TextDataset(df, df[df['fold_group'] == 0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

configuration = BertConfig(max_position_embeddings=128)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=df['label_group'].nunique())
model.to(device)
model.train()

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

for epoch in range(3):
    for input_ids, attention_masks, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids.to(device),
                        attention_mask=attention_masks.to(device),
                        labels=labels.to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=3,              # total # of training epochs
#     per_device_train_batch_size=16,  # batch size per device during training
#     per_device_eval_batch_size=64,   # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
# )
#
# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=train_dataset,         # training dataset
#     eval_dataset=valid_dataset            # evaluation dataset
# )
