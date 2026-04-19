"""
Preprocessing, evaluation, and plotting helpers.
"""

from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TaskType = Literal["binary", "multiclass", "regression"]


def _label_encode_non_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

# ----- Adult (binary classification, labels +1 / -1) -----


def preprocess_adult(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Encode categoricals; labels in {+1, -1} for >50K vs <=50K."""
    df = df.dropna()
    df["income"] = df["income"].str.strip().str.rstrip(".")
    df["label"] = df["income"].apply(lambda x: 1 if x == ">50K" else -1)
    X_df = df.drop(columns=["income", "label"])
    y = df["label"].values.astype(int)
    X_df = _label_encode_non_numeric(X_df)
    X = X_df.values.astype(float)
    return X, y


# ----- Communities and Crime (regression) -----


def preprocess_communities_crime(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    UCI Communities and Crime: skip columns 0-4; features are numeric columns 5..end-1;
    target is last column ViolentCrimesPerPop (real-valued, normalized in UCI).
    """
    df = df.replace("?", np.nan)
    num = df.iloc[:, 5:].apply(pd.to_numeric, errors="coerce")
    num = num.drop(columns=num.columns[num.isna().any()]).dropna()
    y = num.iloc[:, -1].values.astype(float)
    X = num.iloc[:, :-1].values.astype(float)
    return X, y


# ----- MNIST (multiclass, labels 0 .. 9) -----


def preprocess_mnist(
    X: np.ndarray, y: np.ndarray, max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Labels in {0, 1, ..., 9}. Optional subsampling."""
    y = y.astype(int)
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]
    return X.astype(np.float64), y


# ----- Regression tabular (Allstate, Sberbank) -----

def preprocess_allstate_claims(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = "loss",
) -> Tuple[np.ndarray, np.ndarray]:
    """Allstate Claims Severity: real-valued loss."""
    if test_df is not None:
        train_df = pd.concat([train_df, test_df], ignore_index=True)
    if target_col not in train_df.columns:
        raise ValueError(f"Column '{target_col}' not found; expected Allstate train.csv schema.")
    train_df = train_df.dropna()
    y_raw = train_df[target_col].astype(float).values
    X_train_df = train_df.drop(columns=[target_col, "id"], errors="ignore")
    X_train_df = _label_encode_non_numeric(X_train_df)
    X = X_train_df.values.astype(float)
    return X, y_raw


def preprocess_sberbank(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    target_col: str = "price_doc",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sberbank: real-valued price_doc."""
    if test_df is not None:
        train_df = pd.concat([train_df, test_df], ignore_index=True)
    if target_col not in train_df.columns:
        raise ValueError(f"Column '{target_col}' not found; expected Sberbank train.csv schema.")
    train_df = train_df.dropna()
    id_like_train = {c for c in train_df.columns if c.lower() in ("id", "timestamp")}
    y_raw = pd.to_numeric(train_df[target_col], errors="coerce").values
    X_train_df = train_df.drop(columns={target_col} | id_like_train, errors="ignore")
    X_train_df = _label_encode_non_numeric(X_train_df)
    X_train_df = X_train_df.apply(pd.to_numeric, errors="coerce")
    X = X_train_df.values.astype(float)
    y_raw = y_raw.astype(float)
    return X, y_raw


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


def evaluate_binary(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, float]:
    """Accuracy and ROC-AUC (margin scores before sign). Labels in {+1, -1} or {0,1}."""
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores)
    return acc, auc


def evaluate_multiclass(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Accuracy & the confusion matrix"""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """MSE, MAE, R^2."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

# Early stopping may happen, thus the x-axis might not be able to reach the number of estimators set by the user
def get_train_err(
    model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    task_type: str,
    method_name: str,
) -> Tuple[list, list]:
    # TODO: return the correct error (MSE for regression, Error for classification) for the given method and task type
    if method_name == "AdaBoost":
        if task_type == "regression":
            train_err, test_err = [], []
            for pred_tr in model.staged_predict(X_train):
                train_err.append(mean_squared_error(y_train, pred_tr))
            for pred_te in model.staged_predict(X_test):
                test_err.append(mean_squared_error(y_test, pred_te))
        else:
            train_err, test_err = model.train_errors, model.test_errors
    else:
        raise ValueError(f"Unknown method: {method_name}")
    return train_err, test_err


# ----- Plotting -----


def plot_error_curves(
    train_errors: list,
    test_errors: Optional[list],
    title: str,
    outfile: str = "adaboost_error_curve.png",
    ylabel: str = "Classification error",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_errors) + 1), train_errors, label="Train", linewidth=2)
    if test_errors:
        plt.plot(
            range(1, len(test_errors) + 1),
            test_errors,
            label="Test",
            linewidth=2,
            linestyle="--",
        )
    plt.xlabel("Boosting round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.abspath(outfile)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")
