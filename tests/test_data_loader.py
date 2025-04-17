import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pandas as pd
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor


def test_data_loader_real():
    orig_train = os.path.join("data", "processed", "train.csv")
    orig_test = os.path.join("data", "processed", "test.csv")

    df_train = pd.read_csv(orig_train, nrows=200)
    df_test = pd.read_csv(orig_test, nrows=100)

    with tempfile.TemporaryDirectory() as tmp:
        df_train.to_csv(os.path.join(tmp, "train.csv"), index=False)
        df_test.to_csv(os.path.join(tmp, "test.csv"), index=False)

        loader = DataLoader(data_dir=tmp)
        loader.load_data("train.csv", "test.csv")
        train, valid, test = loader.train_valid_split(0.2)

        # Train dataset should remain the same, valid and test should be split
        assert train.shape[0] == 200
        assert valid.shape[0] == 20
        assert test.shape[0] == 80


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


def test_preprocessor():

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

    assert X.shape[0] == train.shape[0]
    assert y.shape[0] == train.shape[0]

    assert X.shape[1] == 9


def main():
    try:
        test_data_loader_real()
    except Exception as e:
        print(f"Test failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("All DataLoader & Preprocessor tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
