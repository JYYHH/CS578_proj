"""
Per-sample loss functions for stability experiments.
All functions take (y_true, y_pred) as 1-D numpy arrays and return a 1-D array of losses.
"""

import numpy as np


def zero_one(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_pred != y_true).astype(float)


def squared(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_pred - y_true) ** 2


def absolute(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_pred - y_true)


def hinge(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """y_true in {-1, +1}, y_pred is the raw margin (pre-threshold)."""
    return np.maximum(0.0, 1.0 - y_true * y_pred)
