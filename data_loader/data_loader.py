import os
import cv2
import pandas as pd
import torchvision.transforms as transform
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CustomDataLoader(Dataset):
    def __init__(
            self,
            csv_path,
            image_transformer=None
    ):
        self.image_df = pd.read_csv(csv_path)
        self.image_df = self.image_df[self.image_df.columns[:-1]]
        self.image_transformer = image_transformer

    def __len__(self):
        return self.image_df.shape[0]

    def __getitem__(self, idx):
        image = self.image_df.iloc[idx].to_numpy().reshape((28, 28))
        if self.image_transformer:
            image = self.image_transformer(image)
        return image


class LoadData:
    def __init__(self, csv_path, batch_size=16):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.image_transformer = transform.Compose(
            [

                transform.ToTensor(),
            ]
        )

    def load_data(self):
        dataset = CustomDataLoader(
            csv_path=self.csv_path,
            image_transformer=self.image_transformer
        )
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size
        )
        return data_loader
