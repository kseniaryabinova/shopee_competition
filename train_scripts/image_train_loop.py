import os
import pandas as pd

import albumentations as alb
import torch
from albumentations.pytorch import ToTensorV2

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import wandb

from dataset import ImageDataset
from model import EfficientNetArcFace

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'

wandb.init(project='shopee_effnet0', group=wandb.util.generate_id())

batch_size = 32
width_size = 128
init_lr = 1e-3
n_epochs = 10
emb_size = 512
margin = 0.5
wandb.config.batch_size = batch_size
wandb.config.width_size = width_size
wandb.config.init_lr = init_lr
wandb.config.n_epochs = n_epochs
wandb.config.emb_size = emb_size

df = pd.read_csv('../dataset/train.csv')
transforms = alb.Compose([
    alb.Resize(width_size, width_size),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
dataset = ImageDataset(df, '../dataset/train_images', transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

model = EfficientNetArcFace(emb_size, df['label_group'].nunique(), backbone='tf_efficientnet_b0_ns',
                            pretrained=True, margin=margin)
model.cuda()

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

model.train()
optimizer.zero_grad()

train_loss = 0

for epoch in range(n_epochs):
    for images, labels in dataloader:
        outputs = model(images.cuda(), labels.cuda())

        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        print(loss.item())

    train_loss /= len(list(iter(dataloader)))


