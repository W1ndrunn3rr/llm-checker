import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class CSVDataLoader(Dataset):
    def __init__(self, file_name, transform=None):
        self.data_frame = pd.read_csv(file_name)
        self.trainsform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        features = self.data_frame.iloc[idx, :-1].values.astype("float32")
        label = self.data_frame.iloc[idx, -1]

        sample = {"features": features, "label": label}

        if self.trainsform:
            sample = self.trainsform(sample)

        return sample["features"], sample["label"]
