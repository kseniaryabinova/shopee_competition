import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import AutoTokenizer


def dummy_df():
    df = pd.DataFrame({
        'text': ['kek', 'lol', 'i want to chill'],
        'label': [0, 1, 2]
    })
    return df


class ClassificationDataset(Dataset):
    def __init__(self, df, max_len):
        self.df = df
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence",
            max_position_embeddings=max_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index]['text']
        tokens = self.tokenizer(text, return_tensors='pt', padding='max_length',
                                max_length=self.max_len)
        label = self.df.iloc[index]['label']

        return torch.squeeze(tokens.input_ids), label
