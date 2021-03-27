import os

import cv2

from torch.utils.data import Dataset
import albumentations as alb


class ImageDataset(Dataset):
    def __init__(self, df, dir_path, transform):
        self.df = df
        self.dir_path = dir_path
        self.transform = transform
        self.classes = self.df['label_group'].unique()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_filename = self.df.iloc[index]['image']
        image_filepath = os.path.join(self.dir_path, image_filename)
        image = cv2.imread(image_filepath)

        if self.transform:
            image = self.transform(image=image)['image']

        label = (self.classes == self.df.iloc[index]['label_group']).astype(int)

        return image, label


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('../dataset/train.csv')
    transforms = alb.Compose([
        alb.Resize(128, 128),
    ])

    dataset = ImageDataset(df, '../dataset/train_images', transforms)

    for i in range(10):
        img, _ = dataset[i]
        cv2.imshow('df', img)
        if cv2.waitKey(0) == 27:
            break

