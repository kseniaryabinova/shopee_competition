import os
from pytz import timezone
from datetime import datetime

import numpy as np
import pandas as pd

import albumentations as alb
from albumentations.pytorch import ToTensorV2

from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss, SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import wandb

from dataset import ImageDataset
from model import EfficientNetArcFace
from train_functions import train_one_epoch, evaluate


def train_function(gpu, world_size, node_rank, gpus):
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

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    batch_size = 64
    width_size = 416
    init_lr = 1e-4
    end_lr = 1e-6
    n_epochs = 20
    emb_size = 512
    margin = 0.5
    dropout = 0.0
    iters_to_accumulate = 10

    checkpoints_dir_name = 'effnet4_{}_{}'.format(width_size, dropout)
    os.makedirs(checkpoints_dir_name, exist_ok=True)

    if rank == 0:
        wandb.init(project='shopee_effnet4', group=wandb.util.generate_id())

        wandb.config.model_name = checkpoints_dir_name
        wandb.config.batch_size = batch_size
        wandb.config.width_size = width_size
        wandb.config.init_lr = init_lr
        wandb.config.n_epochs = n_epochs
        wandb.config.emb_size = emb_size
        wandb.config.dropout = dropout
        wandb.config.iters_to_accumulate = iters_to_accumulate
        wandb.config.optimizer = 'adam'
        wandb.config.scheduler = 'CosineAnnealingLR'

    df = pd.read_csv('../../dataset/folds.csv')
    train_df = df[df['fold'] != 0]
    transforms = alb.Compose([
        # alb.RandomResizedCrop(width_size, width_size),
        # alb.HorizontalFlip(),
        # alb.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30),
        # alb.CoarseDropout(max_height=int(width_size*0.1), max_width=int(width_size*0.1)),
        alb.Resize(width_size, width_size),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    train_set = ImageDataset(df, train_df, '../../dataset/train_images', transforms)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_set, batch_size=batch_size // world_size, shuffle=False, num_workers=4,
                                  sampler=sampler)

    valid_df = df[df['fold'] == 0]
    transforms = alb.Compose([
        alb.Resize(width_size, width_size),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    valid_set = ImageDataset(df, valid_df, '../../dataset/train_images', transforms)
    valid_dataloader = DataLoader(valid_set, batch_size=batch_size // world_size, shuffle=False, num_workers=4)

    model = EfficientNetArcFace(emb_size, df['label_group'].nunique(), device, dropout=dropout,
                                backbone='tf_efficientnet_b4_ns', pretrained=True, margin=margin, is_amp=True)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[gpu])

    scaler = GradScaler()
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=end_lr, last_epoch=-1)

    model.train()
    optimizer.zero_grad()

    for epoch in range(n_epochs):
        train_loss, train_duration, train_f1 = train_one_epoch(model, train_dataloader, optimizer, criterion, device,
                                                               scaler, iters_to_accumulate=iters_to_accumulate)
        scheduler.step()

        if rank == 0:
            valid_loss, valid_duration, valid_f1 = evaluate(model, valid_dataloader, criterion, device)

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

    if rank == 0:
        wandb.finish()
