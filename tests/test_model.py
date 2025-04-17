import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.model.model import FraudDetectionModel
import numpy as np


def detect_date_columns(df):
    date_columns = []
    for col in df.columns:
        try:
            temp_series = pd.to_datetime(df[col], errors="coerce")
            if temp_series.notna().any():
                date_columns.append(col)
        except Exception:
            pass
    return date_columns


def test_model() -> float:

    d = DataLoader("data/processed")

    d.load_data("train.csv", "test.csv")

    train, val, test = d.train_valid_split(0.2)

    date_columns = detect_date_columns(train.copy())

    categorical = [
        col
        for col in train.select_dtypes(include=["object"]).columns
        if col not in date_columns
    ]
    numerical = train.select_dtypes(include=["float64", "int64"]).columns

    p = DataPreprocessor(categorical, numerical, "is_fraud", date_columns)

    p = p.fit(train)
    X, y = p.transform(train)
    y = np.array(y).astype(np.float64)

    fd = FraudDetectionModel(
        X.shape[1],
    )

    fd.train(X, y)
    ans = fd.evaluate(X, y)

    return ans["accuracy"]


def main():
    accuracy = -1
    try:
        accuracy = test_model()
    except Exception as e:
        print(f"Test failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Regression test
    if accuracy < 0.8:
        print("Model accuracy too low:", accuracy, file=sys.stderr)
        sys.exit(1)

    print("All model tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
