import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import LabelEncoder


# ============================================================
# UCI Adult Income Dataset
# ============================================================

def load_adult_dataset():
    """Downloads and returns the UCI Adult Income dataset as a DataFrame."""
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    col_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    print("Downloading Adult dataset...")
    train_df = pd.read_csv(url_train, header=None, names=col_names,
                           na_values=" ?", skipinitialspace=True)
    test_df  = pd.read_csv(url_test,  header=None, names=col_names,
                           na_values=" ?", skipinitialspace=True, skiprows=1)

    return pd.concat([train_df, test_df], ignore_index=True)


def preprocess_adult(df):
    """
    Cleans and encodes the Adult DataFrame.
    Labels are encoded as +1 (income >50K) / -1 (income <=50K).
    This works for both AdaBoost (requires signed labels) and sklearn estimators.

    Returns:
        X : np.ndarray of shape (n_samples, n_features)
        y : np.ndarray of +1/-1 labels
    """
    df = df.dropna().copy()
    df["income"] = df["income"].str.strip().str.rstrip(".")
    df["label"]  = df["income"].apply(lambda x: 1 if x == ">50K" else -1)

    X_df = df.drop(columns=["income", "label"])
    y    = df["label"].values

    for col in X_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    return X_df.values.astype(float), y


# ============================================================
# California Housing Dataset
# ============================================================

def load_california_housing():
    """
    Loads the California Housing dataset via sklearn.

    Returns:
        X : np.ndarray of shape (n_samples, 8)
        y : np.ndarray of median house values (in $100k units)
    """
    housing = fetch_california_housing()
    return housing.data, housing.target
