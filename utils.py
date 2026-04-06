"""
Preprocessing, evaluation, and plotting helpers.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ----- Adult (UCI) -----


def preprocess_adult(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Missing values, encode categoricals; labels in {+1, -1} for >50K vs <=50K."""
    df = df.dropna()
    df["income"] = df["income"].str.strip().str.rstrip(".")
    df["label"] = df["income"].apply(lambda x: 1 if x == ">50K" else -1)
    X_df = df.drop(columns=["income", "label"])
    y = df["label"].values
    for col in X_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
    X = X_df.values.astype(float)
    return X, y


# ----- Communities and Crime (UCI) -----


def preprocess_communities_crime(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    UCI Communities and Crime: skip non-predictive columns 0-4 (incl. string
    communityname); from column 5 onward all fields are numeric in the UCI file.
    Last column is ViolentCrimesPerPop (normalized). Binary labels via median split.
    """
    df = df.replace("?", np.nan)
    num = df.iloc[:, 5:].apply(pd.to_numeric, errors="coerce").dropna()
    y_cont = num.iloc[:, -1].values.astype(float)
    X = num.iloc[:, :-1].values.astype(float)
    median = np.median(y_cont)
    y = np.where(y_cont > median, 1, -1)
    return X, y


# ----- MNIST (binary: digits 0 vs 1) -----


def preprocess_mnist_binary(
    X: np.ndarray, y: np.ndarray, max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep only digits 0 and 1; labels {-1, +1}. Optional subsampling for speed."""
    y = y.astype(int)
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    y_signed = np.where(y == 0, -1, 1)
    return X.astype(np.float64), y_signed


# ----- Tabular regression CSVs -> binary classification (median split) -----


def _label_encode_objects(df: pd.DataFrame, skip: set) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in skip:
            continue
        if out[col].dtype == object or str(out[col].dtype) == "string":
            out[col] = out[col].fillna("__na__").astype(str)
            out[col] = LabelEncoder().fit_transform(out[col])
    return out


def preprocess_allstate_claims(
    df: pd.DataFrame, target_col: str = "loss"
) -> Tuple[np.ndarray, np.ndarray]:
    """Allstate Claims Severity: median split on loss -> {-1, +1}."""
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found; expected Allstate train.csv schema.")
    drop_cols = {"id"} & set(df.columns)
    y_raw = df[target_col].astype(float).values
    X_df = df.drop(columns={target_col} | drop_cols)
    X_df = _label_encode_objects(X_df, skip=set())
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(float)
    median = np.median(y_raw)
    y = np.where(y_raw > median, 1, -1)
    return X, y


def preprocess_sberbank(
    df: pd.DataFrame, target_col: str = "price_doc"
) -> Tuple[np.ndarray, np.ndarray]:
    """Sberbank Russian Housing: median split on price_doc -> {-1, +1}."""
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found; expected Sberbank train.csv schema.")
    id_like = {c for c in df.columns if c.lower() in ("id", "timestamp")}
    y_raw = pd.to_numeric(df[target_col], errors="coerce").values
    X_df = df.drop(columns={target_col} | id_like, errors="ignore")
    X_df = _label_encode_objects(X_df, skip=set())
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    mask = ~np.isnan(y_raw)
    X_df = X_df.loc[mask].reset_index(drop=True)
    y_raw = y_raw[mask]
    X = X_df.values.astype(float)
    median = np.median(y_raw)
    y = np.where(y_raw > median, 1, -1)
    return X, y


# ----- Train / test split -----


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)


# ----- Evaluation -----


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, float]:
    """Accuracy and ROC-AUC (scores are real-valued margins before sign)."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    return acc, auc


# ----- Plotting -----


def plot_error_curves(
    train_errors: list,
    test_errors: Optional[list],
    title: str,
    outfile: str = "adaboost_error_curve.png",
) -> None:
    """Plot training (and optional test) classification error vs. boosting round."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_errors) + 1), train_errors, label="Train Error", linewidth=2)
    if test_errors:
        plt.plot(
            range(1, len(test_errors) + 1),
            test_errors,
            label="Test Error",
            linewidth=2,
            linestyle="--",
        )
    plt.xlabel("Boosting Round")
    plt.ylabel("Classification Error")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.abspath(outfile)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")
