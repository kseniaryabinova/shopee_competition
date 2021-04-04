import torch
import albumentations as alb
from albumentations.pytorch import ToTensorV2

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import EfficientNetArcFace

batch_size = 16
width_size = 192
init_lr = 1e-4
end_lr = 1e-6
n_epochs = 20
emb_size = 512
margin = 0.5

df = pd.read_csv('../dataset/train.csv')

device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
model = EfficientNetArcFace(emb_size, df['label_group'].nunique(), device, dropout=0.,
                            backbone='tf_efficientnet_b4_ns', pretrained=False, margin=margin, is_amp=False)
model.load_state_dict(torch.load('../models/effnet4_epoch19_loss3.249993135680014_f10.9268540912270338.pth'))
model.to(device)
model.eval()

transforms = alb.Compose([
    alb.Resize(width_size, width_size),
    alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

dataset = ImageDataset(df, '../dataset/train_images', transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

embeddings = None

with torch.no_grad():
    for images, labels in dataloader:
        outputs = model(images.cuda())

        if embeddings is None:
            embeddings = outputs.cpu().detach().numpy()
        else:
            embeddings = np.concatenate([embeddings, outputs.cpu().detach().numpy()], axis=0)

np.savez_compressed('../embeddings/embs_effnet4.npz', embeddings=embeddings)
