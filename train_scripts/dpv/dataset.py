import pandas as pd

from torch.utils.data import Dataset
from transformers import AutoTokenizer


def dummy_df():
    df = pd.DataFrame({
        'text': ['kek', 'lol', 'i want to chill'],
        'label': [1, 2, 3]
    })
    return df


class ClassificationDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        tokens = self.tokenizer(text, return_tensors='pt', padding=True)
        label = self.df.iloc[index]['label']

        return tokens, label
