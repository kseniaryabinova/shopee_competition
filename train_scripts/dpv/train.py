from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dpv.dataset import ClassificationDataset, dummy_df
from dpv.model import DeepPavlovBERT

dataset = ClassificationDataset(dummy_df())
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

model = DeepPavlovBERT(3)
optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss()


