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
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

import wandb

from dataset import ImageDataset
from model import EfficientNetArcFace
from train_functions import train_one_epoch, evaluate, get_embeddings, LabelSmoothLoss, validate_embeddings_f1, \
    ShopeeScheduler


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

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    batch_size = 64
    width_size = 416
    init_lr = 1e-4
    end_lr = 1e-6
    n_epochs = 20
    emb_size = 512
    margin = 0.5
    dropout = 0.0
    iters_to_accumulate = 1

    if rank == 0:
        wandb.init(project='shopee_effnet0', group=group_name, job_type=str(fold_number))

        checkpoints_dir_name = 'effnet0_{}_{}_{}'.format(width_size, dropout, group_name)
        os.makedirs(checkpoints_dir_name, exist_ok=True)

        wandb.config.model_name = checkpoints_dir_name
        wandb.config.batch_size = batch_size
        wandb.config.width_size = width_size
        wandb.config.init_lr = init_lr
        wandb.config.n_epochs = n_epochs
        wandb.config.emb_size = emb_size
        wandb.config.dropout = dropout
        wandb.config.iters_to_accumulate = iters_to_accumulate
        wandb.config.optimizer = 'adam'
        wandb.config.scheduler = 'ShopeeScheduler'

    df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
    train_df = df[df['fold_group'] != fold_number]
    train_transforms = alb.Compose([
        alb.RandomResizedCrop(width_size, width_size),
        alb.ShiftScaleRotate(shift_limit=0.1, rotate_limit=30),
        alb.HorizontalFlip(),
        alb.OneOf([
            alb.Sequential([
                alb.HueSaturationValue(hue_shift_limit=50),
                alb.RandomBrightnessContrast(),
            ]),
            alb.FancyPCA(),
            alb.ChannelDropout(),
            alb.ChannelShuffle(),
            alb.RGBShift()
        ]),
        alb.CoarseDropout(max_height=int(width_size*0.1), max_width=int(width_size*0.1)),
        alb.OneOf([
            alb.ElasticTransform(),
            alb.OpticalDistortion(),
            alb.GridDistortion()
        ]),
        alb.Resize(width_size, width_size),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    train_set = ImageDataset(train_df, train_df, '../../dataset/train_images', train_transforms)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_set, batch_size=batch_size // world_size, shuffle=False, num_workers=4,
                                  sampler=sampler)

    # valid_df = df[df['fold_strat'] == fold_number]
    valid_transforms = alb.Compose([
        alb.Resize(width_size, width_size),
        alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # valid_set = ImageDataset(train_df, valid_df, '../../dataset/train_images', valid_transforms)
    # valid_dataloader = DataLoader(valid_set, batch_size=batch_size // world_size, shuffle=False, num_workers=4)

    test_df = df[df['fold_group'] == fold_number]
    test_set = ImageDataset(test_df, test_df, '../../dataset/train_images', valid_transforms)
    test_dataloader = DataLoader(test_set, batch_size=batch_size // world_size, shuffle=False, num_workers=4)

    model = EfficientNetArcFace(emb_size, train_df['label_group'].nunique(), device, dropout=dropout,
                                backbone='tf_efficientnet_b0_ns', pretrained=True, margin=margin, is_amp=True)
    model = SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[gpu])

    scaler = GradScaler()
    criterion = CrossEntropyLoss()
    # criterion = LabelSmoothLoss(smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=end_lr,
    #                               last_epoch=-1)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2000, T_mult=1,
    #                                         eta_min=end_lr, last_epoch=-1)
    scheduler = ShopeeScheduler(optimizer, lr_start=init_lr,
                                lr_max=init_lr*batch_size, lr_min=end_lr)

    for epoch in range(n_epochs):
        train_loss, train_duration, train_f1 = train_one_epoch(
            model, train_dataloader, optimizer, criterion, device, scaler,
            scheduler=None, iters_to_accumulate=iters_to_accumulate)
        scheduler.step()

        if rank == 0:
            # valid_loss, valid_duration, valid_f1 = evaluate(model, valid_dataloader, criterion, device)
            embeddings = get_embeddings(model, test_dataloader, device)
            embeddings_f1 = validate_embeddings_f1(embeddings, test_df)

            wandb.log({'train_loss': train_loss, 'train_f1': train_f1,
                       'embeddings_f1': embeddings_f1, 'epoch': epoch})

            filename = '{}_foldnum{}_epoch{}_train_loss{}_f1{}'.format(
                checkpoints_dir_name, fold_number+1, epoch+1,
                round(train_loss, 3), round(embeddings_f1, 3))
            torch.save(model.module.state_dict(), os.path.join(checkpoints_dir_name, '{}.pth'.format(filename)))
            # np.savez_compressed(os.path.join(checkpoints_dir_name, '{}.npz'.format(filename)), embeddings=embeddings)

            print('FOLD NUMBER %d\tEPOCH %d:\t'
                  'TRAIN [duration %.3f sec, loss: %.3f, avg f1: %.3f]\t'
                  'VALID EMBEDDINGS [avg f1: %.3f]\tCurrent time %s' %
                  (fold_number + 1, epoch + 1, train_duration,
                   train_loss, train_f1, embeddings_f1,
                   str(datetime.now(timezone('Europe/Moscow')))))

    if rank == 0:
        wandb.finish()
