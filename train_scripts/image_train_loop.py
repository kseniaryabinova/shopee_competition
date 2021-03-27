import os
import pandas as pd

import albumentations as alb
from torch.utils.data import DataLoader

import wandb

from dataset import ImageDataset

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':64:8'

wandb.init(project='shopee_effnet0', group=wandb.util.generate_id())

batch_size = 32
width_size = 128
wandb.config.batch_size = batch_size
wandb.config.width_size = width_size

df = pd.read_csv('../dataset/train.csv')
transforms = alb.Compose([
        alb.Resize(width_size, width_size),
    ])
dataset = ImageDataset(df, '../dataset/train_images', transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


