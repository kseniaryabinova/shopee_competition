import re

from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer


class TextDataset(Dataset):
    def __init__(self, df, bert_name='bert-base-multilingual-cased', max_len=128):
        self.df = df
        self.classes = df['label_group'].unique().tolist()

        self.input_ids = []
        self.attention_mask = []

        tokenizer = BertTokenizer.from_pretrained(bert_name, max_position_embeddings=max_len)
        for text in self.df['title']:
            text = text.lower()
            text = re.sub(r"\\x..", r"", text)
            encoding = tokenizer(text, return_tensors='pt', padding='max_length', max_length=max_len, truncation=True)
            self.input_ids.append(encoding['input_ids'].squeeze())
            self.attention_mask.append(encoding['attention_mask'].squeeze())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.classes.index(self.df.iloc[index]['label_group'])
        return self.input_ids[index], self.attention_mask[index], label, index


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
    dataset = TextDataset(df, df[df['fold_group'] != 0])
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=0)
    for input_ids, attention_masks, labels in dataloader:
        pass
