from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from dpv.dataset import ClassificationDataset, dummy_df
from dpv.model import DeepPavlovBERT

dataset = ClassificationDataset(dummy_df(), 32)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = DeepPavlovBERT(3)
optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss()

n_epoch = 10

for epoch in range(n_epoch):
    mean_loss = 0
    iter_counter = 0

    for tokens, labels in dataloader:
        outputs = model(tokens)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        mean_loss += loss.item()
        iter_counter += 1

    print(mean_loss / iter_counter)
