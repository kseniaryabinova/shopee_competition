import re

from torch.utils.data import Dataset

from transformers import BertTokenizer


class TextDataset(Dataset):
    def __init__(self, full_df, df, bert_name='bert-base-multilingual-cased'):
        self.df = df
        self.classes = full_df['label_group'].unique().tolist()

        self.input_ids = []
        self.attention_mask = []

        tokenizer = BertTokenizer.from_pretrained(bert_name)
        for text in self.df['title']:
            text = re.sub(r"\\x..", r"", text)
            encoding = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            self.input_ids.append(encoding['input_ids'])
            self.attention_mask.append(encoding['attention_mask'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.classes.index(self.df.iloc[index]['label_group'])
        return self.input_ids[index], self.attention_mask[index], label
