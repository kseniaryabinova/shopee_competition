import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2

import numpy as np
import pandas as pd
from torch.nn import Softmax
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import EfficientNetArcFace

batch_size = 32
width_size = 128
init_lr = 1e-3
n_epochs = 10
emb_size = 512
margin = 0.5

df = pd.read_csv('../dataset/train.csv')

model = EfficientNetArcFace(emb_size, df['label_group'].nunique(), backbone='tf_efficientnet_b0_ns',
                            pretrained=True, margin=margin)
model.load_state_dict(torch.load('../models/effnet0_train_loss11.136467846308912.pth'))
model.cuda()
model.eval()

transforms = alb.Compose([
    alb.Resize(width_size, width_size),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset = ImageDataset(df, '../dataset/train_images', transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

embeddings = None

for images, labels in dataloader:
    outputs = model(images.cuda())

    if embeddings is None:
        embeddings = outputs.cpu().detach().numpy()
    else:
        embeddings = np.concatenate([embeddings, outputs.cpu().detach().numpy()], axis=0)

np.savez_compressed('../embeddings/embs0.npz', embeddings=embeddings)

