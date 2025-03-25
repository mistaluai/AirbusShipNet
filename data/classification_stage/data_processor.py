import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, dataset_root: str, csv_file: str, seed: int):
        self.dataset_root = dataset_root
        self.csv_file = csv_file
        self.seed = seed
        self.csv_path = os.path.join(self.dataset_root, self.csv_file)

    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = self._preprocess_data()
        return self._split_data(df)

    def _preprocess_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df['ImagePath'] = df['ImageId'].apply(lambda image_id: os.path.join(self.dataset_root, 'train_v2', image_id))
        df['label'] = df['EncodedPixels'].notna().astype(int)
        df.drop(columns=['ImageId', 'EncodedPixels'], inplace=True)
        return df

    def _split_data(self, df: pd.DataFrame):
        train, evaluation = train_test_split(
            df,
            test_size=0.2,
            random_state=self.seed,
            stratify=df['label']
        )
        val, test = train_test_split(
            evaluation,
            test_size=0.5,
            random_state=self.seed,
            stratify=evaluation['label']
        )
        return train, val, test