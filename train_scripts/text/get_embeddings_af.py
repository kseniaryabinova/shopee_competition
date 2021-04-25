import pandas as pd
from torch.utils.data import DataLoader

import torch

from text.dataset import TextDataset
from text.model import BERTWithArcFace

df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
max_len = 128

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = TextDataset(df[df['fold_group'] == 0], max_len=max_len)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

model = BERTWithArcFace(8811)
model.load_state_dict(torch.load('best_bt_af.pth'))
model.float()
model.to(device)
model.eval()

with torch.no_grad():
    for input_ids, attention_masks, labels in dataloader:
        outputs = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_masks.to(device),
        )
        pass
