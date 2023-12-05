import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset


class SingleLabelDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.classes = dataframe.columns[1:].tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # filename column
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Extract labels and convert to ToTensor
        labels = self.dataframe.iloc[idx, 1:].values
        labels = torch.from_numpy(labels.astype('float32'))

        label_index = torch.nonzero(labels, as_tuple=True)[0].item()

        return image, label_index


class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.classes = dataframe.columns[1:].tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # filename column
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        # Extract labels and convert to tensor
        # labes should be idex of column where value = 1
        labels = self.dataframe.iloc[idx, 1:].values
        labels = torch.from_numpy(labels.astype('int32'))

        return image, labels


def build_dataframes(phase_csvs, filters, limit):
    # Read CSV File
    dataframes = {}

    if filters is None:
        filters = dataframes[phase_csvs[0]].columns[1:].tolist()

    for (phase, csv) in phase_csvs.items():
        dataframes[phase] = pd.read_csv(csv)

        dataframes[phase] = dataframes[phase][dataframes[phase][filters].sum(axis=1) > 0]

        if limit is not None:
            dataframes[phase] = dataframes[phase].groupby(filters).head(limit)

    return dataframes

def count_classes(dataframe):
    # omit classes with 0 elements
    return dataframe.iloc[:, 1:].sum(axis=0).filter(lambda x: x > 0)
