import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, dataset_root: str, csv_file: str, seed: int, image_subdir: str = 'train_v2'):
        self.dataset_root = dataset_root
        self.csv_file = csv_file
        self.seed = seed
        self.image_subdir = image_subdir
        self.csv_path = os.path.join(self.dataset_root, self.csv_file)

    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self._split_data(self._preprocess_and_group())

    def _preprocess_and_group(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.dropna(subset=['EncodedPixels'], inplace=True)
        image_dir = os.path.join(self.dataset_root, self.image_subdir)
        df['ImageId'] = df['ImageId'].apply(lambda image_id: os.path.join(image_dir, image_id))

        # Group RLEs by ImageId
        grouped_df = df.groupby('ImageId')['EncodedPixels'].apply(list).reset_index()
        return grouped_df

    def _split_data(self, df: pd.DataFrame):
        train, evaluation = train_test_split(
            df,
            test_size=0.2,
            random_state=self.seed,
            shuffle=True
        )
        val, test = train_test_split(
            evaluation,
            test_size=0.5,
            random_state=self.seed,
        )
        return train, val, test