import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Optional


class DataPreprocessor:
    """
    A class for preprocessing the fraud detection dataset.

    This class handles:
    - Feature engineering
    - Missing value handling
    - Categorical encoding
    - Feature scaling
    - Data imbalance handling
    """

    def __init__(
        self,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        target_column: str = None,
        datetime_features: List[str] = None,
        drop_columns: List[str] = None,
    ):
        """
        Initialize the DataPreprocessor.

        Args:
            categorical_features: List of categorical feature column names
            numerical_features: List of numerical feature column names
            target_column: Name of the target column
            datetime_features: List of datetime feature columns
            drop_columns: List of columns to drop during preprocessing
        """
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.target_column = target_column or ""
        self.datetime_features = datetime_features or ["TransactionDate"]
        self.drop_columns = drop_columns or []

        # Preprocessing objects
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.fitted = False

        # Store column info
        self.encoded_feature_names = []

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with handled missing values
        """
        # Numerical columns: fill with median
        for col in self.numerical_features:
            if col in df and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # Categorical columns: fill with most frequent value
        for col in self.categorical_features:
            if col in df and df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        return df

    def _process_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract useful features from datetime columns.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with extracted datetime features
        """
        df = df.copy()

        for col in self.datetime_features:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

                # Extract useful components
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_year"] = df[col].dt.year

                # Time-based features
                df[f"{col}_is_weekend"] = df[col].dt.dayofweek >= 5

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features that might be useful for fraud detection.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        if "CustomerID" in df.columns and "Amount" in df.columns:
            # Calculate customer statistics
            customer_stats = (
                df.groupby("CustomerID")["Amount"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            customer_stats.columns = [
                "CustomerID",
                "customer_avg_amount",
                "customer_std_amount",
                "customer_tx_count",
            ]

            # Merge back
            df = pd.merge(df, customer_stats, on="CustomerID", how="left")

        return df

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit the preprocessor to the data.

        Args:
            df: Training dataframe to fit the preprocessor on

        Returns:
            Self for method chaining
        """
        df_copy = df.copy()

        # Handle missing values
        df_copy = self._handle_missing_values(df_copy)

        # Process datetime features
        df_copy = self._process_datetime_features(df_copy)

        # Engineer features
        df_copy = self._engineer_features(df_copy)

        # Identify features if not specified
        if not self.categorical_features and not self.numerical_features:
            self.categorical_features = df_copy.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            self.numerical_features = df_copy.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()

            # Remove target from feature lists
            if self.target_column in self.categorical_features:
                self.categorical_features.remove(self.target_column)
            if self.target_column in self.numerical_features:
                self.numerical_features.remove(self.target_column)

        # Fit encoder on categorical features
        if self.categorical_features:
            self.encoder.fit(df_copy[self.categorical_features])
            self.encoded_feature_names = self.encoder.get_feature_names_out(
                self.categorical_features
            ).tolist()

        # Fit scaler on numerical features
        if self.numerical_features:
            self.scaler.fit(df_copy[self.numerical_features])

        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform the data using the fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Tuple of (X, y) where X is the processed features and y is the target (if present)
        """
        if not self.fitted:
            raise ValueError("DataPreprocessor must be fitted before transform")

        df_copy = df.copy()

        # Handle missing values
        df_copy = self._handle_missing_values(df_copy)

        # Process datetime features
        df_copy = self._process_datetime_features(df_copy)

        # Engineer features
        df_copy = self._engineer_features(df_copy)

        # Extract target if present
        y = None
        if self.target_column in df_copy.columns:
            y = df_copy[self.target_column].values

        # Drop unnecessary columns
        for col in self.drop_columns + self.datetime_features:
            if col in df_copy.columns:
                df_copy = df_copy.drop(columns=[col])

        # Transform categorical features
        X_categorical = None
        if self.categorical_features:
            cat_cols = [
                col for col in self.categorical_features if col in df_copy.columns
            ]
            if cat_cols:
                X_categorical = self.encoder.transform(df_copy[cat_cols])

        # Transform numerical features
        X_numerical = None
        if self.numerical_features:
            num_cols = [
                col for col in self.numerical_features if col in df_copy.columns
            ]
            if num_cols:
                X_numerical = self.scaler.transform(df_copy[num_cols])

        # Combine features
        X_parts = []
        if X_categorical is not None and X_categorical.size > 0:
            X_parts.append(X_categorical)
        if X_numerical is not None and X_numerical.size > 0:
            X_parts.append(X_numerical)

        if X_parts:
            X = np.hstack(X_parts)
        else:
            X = np.array([])

        return X, y

    def fit_transform(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit the preprocessor and transform the data.

        Args:
            df: DataFrame to fit and transform

        Returns:
            Tuple of (X, y) where X is the processed features and y is the target (if present)
        """
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """
        Get names of transformed features.

        Returns:
            List of feature names after transformation
        """
        if not self.fitted:
            raise ValueError(
                "DataPreprocessor must be fitted before getting feature names"
            )

        feature_names = []

        # Add encoded categorical features
        if self.encoded_feature_names:
            feature_names.extend(self.encoded_feature_names)

        # Add numerical features
        feature_names.extend(self.numerical_features)

        return feature_names
