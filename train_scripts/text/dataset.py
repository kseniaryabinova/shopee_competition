import re

from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer


class TextDataset(Dataset):
    def __init__(self, full_df, df, bert_name='bert-base-multilingual-cased'):
        self.df = df
        self.classes = full_df['label_group'].unique().tolist()

        self.input_ids = []
        self.attention_mask = []

        tokenizer = BertTokenizer.from_pretrained(bert_name, max_length=128, padding=128)
        for text in self.df['title']:
            text = re.sub(r"\\x..", r"", text)
            encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
            self.input_ids.append(encoding['input_ids'])
            self.attention_mask.append(encoding['attention_mask'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.classes.index(self.df.iloc[index]['label_group'])
        return self.input_ids[index], self.attention_mask[index], label


def collate_fn(data):
    pass


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../../dataset/reliable_validation_tm.csv')
    dataset = TextDataset(df, df[df['fold_group'] != 0])
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for input_ids, attention_masks, labels in dataloader:
        pass
