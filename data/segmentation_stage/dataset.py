import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.tv_tensors import Mask

from PIL import Image, UnidentifiedImageError

## Example Usage https://www.kaggle.com/code/mohamedmohiey/segmentation-datasetusage
class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, original_shape: tuple[int, int] = (768, 768), transform=None):
        self.df = df
        self.original_shape = original_shape
        self.transform = transform or v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
        ]
    )
        self.image_dtype_transform = v2.ToDtype(torch.float32, scale=True)
        self.mask_dtype_transform = v2.ToDtype(torch.float32, scale=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, rle_list = self.df.iloc[idx]
        combined_mask = torch.zeros(self.original_shape, dtype=torch.uint8)

        for rle_mask_str in rle_list:
            single_mask = self.rle_decode(rle_mask_str)
            combined_mask = torch.logical_or(combined_mask, single_mask.to(combined_mask.device))

        mask = Mask(combined_mask.to(torch.uint8))

        try:
            image = Image.open(image_path).convert('RGB')
            image, mask = self.transform(image, mask)

            image = self.image_dtype_transform(image)
            mask = self.mask_dtype_transform(mask)
        except (OSError, UnidentifiedImageError):
            next_idx = idx + 1
            return self.__getitem__(next_idx)

        return image, mask

    @staticmethod
    def rle_decode(mask_rle: str, shape: tuple[int, int] = (768, 768)) -> torch.Tensor:
        s_np = np.asarray(mask_rle.split(), dtype=int)     

        starts = torch.from_numpy(s_np[0::2] - 1) 
        lengths = torch.from_numpy(s_np[1::2])
        ends = starts + lengths

        img_size = shape[0] * shape[1]

        temp_ary = torch.zeros(img_size + 1, dtype=torch.int16)
        # index_add_(dim, index, tensor) -> adds tensor elements to self at indices in index
        temp_ary.index_add_(0, starts, torch.ones_like(starts, dtype=torch.int16))
        temp_ary.index_add_(0, ends, torch.full_like(ends, -1, dtype=torch.int16)) # Add -1 at ends

        # Compute cumulative sum and reshape
        flat_mask = torch.cumsum(temp_ary, dim=0)[:-1] # Remove the extra element

        # Reshape to (W, H) then transpose to (H, W)
        mask = flat_mask.reshape((shape[1], shape[0])).T

        # Return the mask as a tensor of type uint8
        return (mask > 0).to(torch.uint8) 