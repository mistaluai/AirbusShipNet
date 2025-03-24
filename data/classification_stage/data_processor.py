import os
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, dataset_root: str, csv_file: str, seed: int):
        self.dataset_root = dataset_root
        self.csv_file = csv_file
        self.seed = seed
        self.csv_path = os.path.join(self.dataset_root, self.csv_file)
        self.df = self.__preprocess_data()
        self.train_data, self.val_data, self.test_data = self.__split_data()

    def __preprocess_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df['ImagePath'] = df['ImageId'].apply(lambda id: os.path.join(self.dataset_root, 'train_v2', id))
        df['label'] = df['EncodedPixels'].notna().astype(int)
        df.drop(columns=['ImageId', 'EncodedPixels'], inplace=True)
        return df

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.train_data, self.val_data, self.test_data

    def __split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, evaluation = train_test_split(
            self.df,
            test_size=0.2,
            random_state=self.seed,
            stratify=self.df['label']
        )
        val, test = train_test_split(
            evaluation,
            test_size=0.5,
            random_state=self.seed,
            stratify=evaluation['label']
        )
        return train, val, test