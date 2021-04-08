from pytz import timezone
from datetime import datetime
import os
import pandas as pd

import albumentations as alb
import torch
from albumentations.pytorch import ToTensorV2

from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb

from dataset import ImageDataset
from model import EfficientNetArcFace
from train_functions import train_one_epoch, evaluate

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['WANDB_SILENT'] = 'true'

wandb.init(project='shopee_effnet0', group=wandb.util.generate_id())

batch_size = 64
width_size = 416
init_lr = 1e-4
end_lr = 1e-6
n_epochs = 20
emb_size = 512
margin = 0.5
dropout = 0.0
iters_to_accumulate = 10
wandb.config.batch_size = batch_size
wandb.config.width_size = width_size
wandb.config.init_lr = init_lr
wandb.config.n_epochs = n_epochs
wandb.config.emb_size = emb_size
wandb.config.dropout = dropout
wandb.config.iters_to_accumulate = iters_to_accumulate
wandb.config.optimizer = 'adam'
wandb.config.scheduler = 'CosineAnnealingLR'

checkpoints_dir_name = 'effnet4_{}_{}'.format(width_size, dropout)
os.makedirs(checkpoints_dir_name, exist_ok=True)
wandb.config.model_name = checkpoints_dir_name

df = pd.read_csv('../../dataset/folds.csv')
train_df = df[df['fold'] != 0]
transforms = alb.Compose([
    alb.Resize(width_size, width_size),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
dataset = ImageDataset(train_df, '../../dataset/train_images', transforms)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

valid_df = df[df['fold'] == 0]
transforms = alb.Compose([
    alb.Resize(width_size, width_size),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])
valid_set = ImageDataset(valid_df, '../../dataset/train_images', transforms)
valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetArcFace(emb_size, df['label_group'].nunique(), device, dropout=dropout,
                            backbone='tf_efficientnet_b4_ns', pretrained=True, margin=margin, is_amp=True)
model.to(device)

scaler = GradScaler()
model = DataParallel(model)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=end_lr, last_epoch=-1)

model.train()
optimizer.zero_grad()

for epoch in range(n_epochs):
    train_loss, train_duration, train_f1 = train_one_epoch(model, train_dataloader, optimizer, criterion, device,
                                                           scaler, iters_to_accumulate=iters_to_accumulate)
    scheduler.step()

    # valid_loss, valid_duration, valid_f1 = evaluate(model, valid_dataloader, criterion, device)

    wandb.log({'train_loss': train_loss, 'train_f1': train_f1,
               'valid_loss': valid_loss, 'valid_f1': valid_f1, 'epoch': epoch})

    print('EPOCH %d:\tTRAIN [duration %.3f sec, loss: %.3f, avg f1: %.3f]\t\t'
          'VALID [duration %.3f sec, loss: %.3f, avg f1: %.3f]\t\tCurrent time %s' %
          (epoch + 1, train_duration, train_loss, train_f1, valid_duration, valid_loss, valid_f1,
           str(datetime.now(timezone('Europe/Moscow')))))
    torch.save(model.module.state_dict(),
               os.path.join(checkpoints_dir_name, '{}_epoch{}_train_loss{}_f1{}_valid_loss{}_f1().pth'.format(
                   checkpoints_dir_name, epoch, round(train_loss, 3), round(train_f1, 3),
                   round(valid_loss, 3), round(valid_f1, 3))))

wandb.finish()
