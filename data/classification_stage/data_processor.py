import os
import pandas as pd

class DataProcessor:
    def __init__(self, dataset_root: str, csv_file: str):
        self.dataset_root = dataset_root
        self.csv_file = csv_file
        self.csv_path = os.path.join(self.dataset_root, self.csv_file)

    def preprocess_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df['ImagePath'] = df['ImageId'].apply(lambda id: os.path.join(self.dataset_root, 'train_v2', id))
        df['label'] = df['EncodedPixels'].notna().astype(int)
        df.drop(columns=['ImageId', 'EncodedPixels'], inplace=True)
        return df
