import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.transforms import v2

class ClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform or v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx]['ImagePath']
        label = self.df.iloc[idx]['label']

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label

