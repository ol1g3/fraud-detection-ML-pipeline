import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


class DataLoader:
    """
    A class for loading and managing fraud detection datasets.

    This class handles:
    - Loading data from various sources
    - Training/validation/test splits
    - Batch generation for training
    """

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the DataLoader.

        Args:
            data_dir: Directory containing the datasets
        """
        self.data_dir = data_dir
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filename: Name of the file to load

        Returns:
            Loaded DataFrame
        """
        file_path = os.path.join(self.data_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        return df

    def load_data(
        self,
        filename: str,
        test_filename: str,
        sample_frac: float,
        random_state: int = 42,
    ) -> None:
        """
        Load the main dataset.

        Args:
            filename: Name of the training file
            test_filename: Name of the test file (if separate)
            sample_frac: Fraction of data to sample
            random_state: Random state for reproducibility

        """
        self.train_data = self.load_csv(filename, sample_frac, random_state)
        self.test_data = self.load_csv(test_filename, sample_frac, random_state)

    def train_valid_split(
        self,
        df: pd.DataFrame,
        valid_size: float = 0.15,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            df: DataFrame to split
            target_column: Name of the target column for stratification
            train_size: Fraction of data for training
            valid_size: Fraction of data for validation
            random_state: Random state for reproducibility
            stratify: Whether to use stratified splits

        Returns:
            tuple of (train, validation, test) DataFrames
        """

        valid_df, test_df = train_test_split(
            self.test_data,
            test_size=valid_size,
            random_state=random_state,
        )

        self.valid_data = valid_df
        self.test_data = test_df

        return self.train_df, valid_df, test_df

    def get_batch(
        self, dataset: str = "train", batch_size: int = 32, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Get a batch of data for training or evaluation.

        Args:
            dataset: Which dataset to use ('train', 'valid', 'test')
            batch_size: Size of the batch
            shuffle: Whether to shuffle the data
            random_state: Random state for reproducibility

        Returns:
            Batch as DataFrame
        """
        if dataset == "train" and self.train_data is not None:
            data = self.train_data
        elif dataset == "valid" and self.valid_data is not None:
            data = self.valid_data
        elif dataset == "test" and self.test_data is not None:
            data = self.test_data
        else:
            raise ValueError(f"Dataset '{dataset}' not loaded or not recognized")

        batch = data.sample(n=min(batch_size, len(data)), random_state=random_state)

        return batch

    def batch_generator(
        self,
        dataset: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generate batches of data for training or evaluation.

        Args:
            dataset: Which dataset to use ('train', 'valid', 'test')
            batch_size: Size of the batch
            shuffle: Whether to shuffle the data
            random_state: Random state for reproducibility

        Yields:
            Batches of data as DataFrames
        """
        if dataset == "train" and self.train_data is not None:
            data = self.train_data
        elif dataset == "valid" and self.valid_data is not None:
            data = self.valid_data
        elif dataset == "test" and self.test_data is not None:
            data = self.test_data
        else:
            raise ValueError(f"Dataset '{dataset}' not loaded or not recognized")

        indices = np.arange(len(data))

        if shuffle:
            np.random.seed(random_state)
            np.random.shuffle(indices)

        for start_idx in range(0, len(data), batch_size):
            end_idx = min(start_idx + batch_size, len(data))
            batch_indices = indices[start_idx:end_idx]

            yield data.iloc[batch_indices].reset_index(drop=True)
