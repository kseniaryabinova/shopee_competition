from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, full_df, df):
        self.df = df
        self.classes = full_df['label_group'].unique().tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df.iloc[index, 'title']
        label = self.classes.index(self.df.iloc[index]['label_group'])
        return text, label
