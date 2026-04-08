"""
Download / load raw data and route through utils preprocessing to (X, y, task).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    TaskType,
    preprocess_adult,
    preprocess_allstate_claims,
    preprocess_communities_crime,
    preprocess_mnist,
    preprocess_sberbank,
)

# UCI Adult
ADULT_TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

# UCI Communities and Crime (dataset 183)
COMMUNITIES_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
)


def _default_kaggle_dir() -> Path:
    return Path(__file__).resolve().parent / "data" # "./data/"


def load_adult() -> Tuple[np.ndarray, np.ndarray, TaskType]:
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]
    print("Loading UCI Adult (download if needed)...")
    train_df = pd.read_csv(
        ADULT_TRAIN_URL,
        header=None,
        names=col_names,
        na_values=" ?",
        skipinitialspace=True,
    )
    test_df = pd.read_csv(
        ADULT_TEST_URL,
        header=None,
        names=col_names,
        na_values=" ?",
        skipinitialspace=True,
        skiprows=1,
    )
    df = pd.concat([train_df, test_df], ignore_index=True)
    X, y = preprocess_adult(df)
    return X, y, "binary"


def load_communities_crime() -> Tuple[np.ndarray, np.ndarray, TaskType]:
    print("Loading UCI Communities and Crime (download if needed)...")
    df = pd.read_csv(COMMUNITIES_URL, header=None, na_values="?")
    X, y = preprocess_communities_crime(df)
    return X, y, "regression"


def load_mnist(max_samples: Optional[int] = 12000) -> Tuple[np.ndarray, np.ndarray, TaskType]:
    print("Loading MNIST (OpenML, may download once)...")
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    y = y.astype(int)
    X, y = preprocess_mnist(X, y, max_samples=max_samples)
    return X, y, "multiclass"


def load_ca_housing() -> Tuple[np.ndarray, np.ndarray, TaskType]:
    """California Housing (sklearn): regression with real-valued median house value."""
    print("Loading California Housing (sklearn, may download once)...")
    from sklearn.datasets import fetch_california_housing

    X, y = fetch_california_housing(return_X_y=True, as_frame=False)
    return X.astype(float), y.astype(float), "regression"


def _resolve_csv_path(env_var: str, default_name: str, kaggle_subdir: str) -> Path:
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    base = _default_kaggle_dir() / kaggle_subdir
    return base / default_name
    # will be "data/<default_name>/"

def load_allstate_claims(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, TaskType]:
    """
    Kaggle: https://www.kaggle.com/c/allstate-claims-severity
    Set ALLSTATE_TRAIN_CSV / ALLSTATE_TEST_CSV
    or place train.csv + test.csv under ${KAGGLE_DATA_DIR}/allstate/
    """
    tr_path = train_path or _resolve_csv_path("ALLSTATE_TRAIN_CSV", "train.csv", "allstate")
    te_path = test_path or _resolve_csv_path("ALLSTATE_TEST_CSV", "test.csv", "allstate")
    if not tr_path.is_file():
        raise FileNotFoundError(
            f"Allstate train.csv not found at {tr_path}. "
            "Download from Kaggle, then set ALLSTATE_TRAIN_CSV or KAGGLE_DATA_DIR/allstate/train.csv."
        )
    if not te_path.is_file():
        raise FileNotFoundError(
            f"Allstate test.csv not found at {te_path}. "
            "Download from Kaggle, then set ALLSTATE_TEST_CSV or KAGGLE_DATA_DIR/allstate/test.csv."
        )
    print(f"Loading Allstate Claims train/test from {tr_path} and {te_path}...")
    train_df = pd.read_csv(tr_path, on_bad_lines='skip').dropna()
    test_df = pd.read_csv(te_path, on_bad_lines='skip').dropna()
    X, y = preprocess_allstate_claims(train_df, test_df=test_df)
    return X, y, "regression"


def load_sberbank(
    train_path: Optional[Path] = None,
    test_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, TaskType]:
    """
    Kaggle Sberbank Russian Housing Market competition data
    (see e.g. https://www.kaggle.com/c/sberbank-russian-housing-market).
    Set SBERBANK_TRAIN_CSV / SBERBANK_TEST_CSV
    or place train.csv + test.csv under ${KAGGLE_DATA_DIR}/sberbank/
    """
    tr_path = train_path or _resolve_csv_path("SBERBANK_TRAIN_CSV", "train.csv", "sberbank")
    te_path = test_path or _resolve_csv_path("SBERBANK_TEST_CSV", "test.csv", "sberbank")
    if not tr_path.is_file():
        raise FileNotFoundError(
            f"Sberbank train.csv not found at {tr_path}. "
            "Download from Kaggle, then set SBERBANK_TRAIN_CSV or KAGGLE_DATA_DIR/sberbank/train.csv."
        )
    if not te_path.is_file():
        raise FileNotFoundError(
            f"Sberbank test.csv not found at {te_path}. "
            "Download from Kaggle, then set SBERBANK_TEST_CSV or KAGGLE_DATA_DIR/sberbank/test.csv."
        )
    print(f"Loading Sberbank train/test from {tr_path} and {te_path}...")
    train_df = pd.read_csv(tr_path, on_bad_lines='skip').dropna()
    test_df = pd.read_csv(te_path, on_bad_lines='skip').dropna()
    X, y = preprocess_sberbank(train_df, test_df=test_df)
    return X, y, "regression"


def load_dataset(
    name: str,
    *,
    mnist_max_samples: Optional[int] = 12000,
    allstate_path: Optional[Path] = None,
    sberbank_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, TaskType]:
    key = name.strip().lower()
    if key in ("adult", "uci_adult"):
        return load_adult()
    if key in ("communities", "communities_crime", "crime", "uci_communities"):
        return load_communities_crime()
    if key in ("mnist", "mnist10", "mnist_multiclass"):
        mnist_max = None if mnist_max_samples == 0 else mnist_max_samples
        return load_mnist(max_samples=mnist_max)
    if key in ("ca_housing", "california_housing", "california"):
        return load_ca_housing()
    if key in ("allstate", "allstate_claims"):
        return load_allstate_claims(allstate_path)
    if key in ("sberbank", "sberbank_housing"):
        return load_sberbank(sberbank_path)
    raise ValueError(
        f"Unknown dataset '{name}'. Choose: adult, communities_crime, mnist, ca_housing, allstate, sberbank."
    )
