"""
Download / load raw data and route through utils preprocessing to (X, y).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    preprocess_adult,
    preprocess_allstate_claims,
    preprocess_communities_crime,
    preprocess_mnist_binary,
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
    return Path(os.environ.get("KAGGLE_DATA_DIR", Path(__file__).resolve().parent / "data"))


def load_adult() -> Tuple[np.ndarray, np.ndarray]:
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
    return preprocess_adult(df)


def load_communities_crime() -> Tuple[np.ndarray, np.ndarray]:
    print("Loading UCI Communities and Crime (download if needed)...")
    df = pd.read_csv(COMMUNITIES_URL, header=None, na_values="?")
    return preprocess_communities_crime(df)


def load_mnist_binary(max_samples: Optional[int] = 12000) -> Tuple[np.ndarray, np.ndarray]:
    print("Loading MNIST (OpenML, may download once)...")
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    y = y.astype(int)
    return preprocess_mnist_binary(X, y, max_samples=max_samples)


def _resolve_csv_path(env_var: str, default_name: str, kaggle_subdir: str) -> Path:
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    base = _default_kaggle_dir() / kaggle_subdir
    return base / default_name


def load_allstate_claims(train_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaggle: https://www.kaggle.com/c/allstate-claims-severity
    Set ALLSTATE_TRAIN_CSV or place train.csv under ${KAGGLE_DATA_DIR}/allstate/
    """
    path = train_path or _resolve_csv_path("ALLSTATE_TRAIN_CSV", "train.csv", "allstate")
    if not path.is_file():
        raise FileNotFoundError(
            f"Allstate train.csv not found at {path}. "
            "Download from Kaggle, then set ALLSTATE_TRAIN_CSV or KAGGLE_DATA_DIR/allstate/train.csv."
        )
    print(f"Loading Allstate Claims from {path}...")
    df = pd.read_csv(path)
    return preprocess_allstate_claims(df)


def load_sberbank(train_path: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaggle Sberbank Russian Housing Market competition data
    (see e.g. https://www.kaggle.com/c/sberbank-russian-housing-market).
    Set SBERBANK_TRAIN_CSV or place train.csv under ${KAGGLE_DATA_DIR}/sberbank/
    """
    path = train_path or _resolve_csv_path("SBERBANK_TRAIN_CSV", "train.csv", "sberbank")
    if not path.is_file():
        raise FileNotFoundError(
            f"Sberbank train.csv not found at {path}. "
            "Download from Kaggle, then set SBERBANK_TRAIN_CSV or KAGGLE_DATA_DIR/sberbank/train.csv."
        )
    print(f"Loading Sberbank from {path}...")
    df = pd.read_csv(path)
    return preprocess_sberbank(df)


def load_dataset(
    name: str,
    *,
    mnist_max_samples: Optional[int] = 12000,
    allstate_path: Optional[Path] = None,
    sberbank_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    key = name.strip().lower()
    if key in ("adult", "uci_adult"):
        return load_adult()
    if key in ("communities", "communities_crime", "crime", "uci_communities"):
        return load_communities_crime()
    if key in ("mnist", "mnist01", "mnist_binary"):
        return load_mnist_binary(max_samples=mnist_max_samples)
    if key in ("allstate", "allstate_claims"):
        return load_allstate_claims(allstate_path)
    if key in ("sberbank", "sberbank_housing"):
        return load_sberbank(sberbank_path)
    raise ValueError(
        f"Unknown dataset '{name}'. Choose: adult, communities_crime, mnist, allstate, sberbank."
    )
