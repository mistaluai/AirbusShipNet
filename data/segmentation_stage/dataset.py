import torch
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import v2

class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, original_shape: tuple[int, int] = (768, 768), transform=None):
        self.df = df
        self.original_shape = original_shape
        self.transform = transform or v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
        ]
    )

    @staticmethod
    def rle_decode(mask_rle: str, shape: tuple[int, int] = (768, 768)) -> torch.Tensor:
        pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, rle_list = self.df.iloc[idx]
        combined_mask = torch.zeros(self.original_shape, dtype=torch.uint8)

        return