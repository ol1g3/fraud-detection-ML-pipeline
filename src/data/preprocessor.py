import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from typing import Optional


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
        categorical_features: list[str] = [],
        numerical_features: list[str] = [],
        target_column: str = "",
        datetime_features: list[str] = [],
        drop_columns: list[str] = [],
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
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target_column = target_column
        self.datetime_features = datetime_features
        self.drop_columns = drop_columns

        # Preprocessing objects
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        self.fitted = False

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
                mode_values = df[col].mode()
                if not mode_values.empty:
                    df[col] = df[col].fillna(mode_values[0])
                else:
                    df[col] = df[col].fillna("")

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

                # Weekend indicator
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

        # Fit encoder on categorical features
        self.encoder.fit(df_copy[self.categorical_features])
        self.encoded_feature_names = self.encoder.get_feature_names_out(
            self.categorical_features
        ).tolist()

        # Fit scaler on numerical features
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

        # Extract target variable
        y = df_copy[self.target_column].values

        # Drop unnecessary columns
        for col in self.drop_columns + self.datetime_features:
            if col in df_copy.columns:
                df_copy = df_copy.drop(columns=[col])

        # Transform categorical features
        X_categorical = None
        cat_cols = [col for col in self.categorical_features if col in df_copy.columns]
        if cat_cols:
            X_categorical = self.encoder.transform(df_copy[cat_cols])

        # Transform numerical features
        X_numerical = None
        num_cols = [col for col in self.numerical_features if col in df_copy.columns]
        if num_cols:
            X_numerical = self.scaler.transform(df_copy[num_cols])

        X_parts = []
        if X_categorical is not None and X_categorical.size > 0:
            X_parts.append(X_categorical)
        if X_numerical is not None and X_numerical.size > 0:
            X_parts.append(X_numerical)

        if X_parts:
            X = np.hstack(X_parts)
        else:
            X = np.array([])

        if len(X[0]) > 1:
            X = X[:, :-1]
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

    def get_feature_names(self) -> list[str]:
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

        if self.encoded_feature_names:
            feature_names.extend(self.encoded_feature_names)

        feature_names.extend(self.numerical_features)

        return feature_names
