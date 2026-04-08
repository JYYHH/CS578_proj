#!/usr/bin/env python3
"""
Train AdaBoost on a chosen dataset and report metrics + error curves.
"""

from __future__ import annotations

import argparse

import numpy as np

from dataset import load_dataset
from ensemble_method import (
    AdaBoostBinaryClassifier,
    AdaBoostMulticlassClassifier,
    AdaBoostRegressor,
    # TODO: for your methods
)
from utils import (
    evaluate_binary,
    evaluate_multiclass,
    evaluate_regression,
    plot_error_curves,
    get_train_err,
    split_train_test,
)

def get_model(method: str, task_type: str, args: argparse.Namespace, class_num: int = 2): 
    # TODO: return the correct model based on the method and task type
    if method == "AdaBoost":
        if task_type == "binary":
            return AdaBoostBinaryClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
        elif task_type == "multiclass":
            return AdaBoostMulticlassClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, class_num=class_num)
        elif task_type == "regression":
            return AdaBoostRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    else:
        raise ValueError(f"Unknown method: {method}")

def parse_args():
    p = argparse.ArgumentParser(description="AdaBoost ensemble on tabular / MNIST data")
    p.add_argument(
        "--dataset",
        type=str,
        default="adult",
        help="adult | communities_crime | mnist | ca_housing | allstate | sberbank",
    )
    p.add_argument(
        "--method", 
        type=str, 
        default="AdaBoost",
        help="AdaBoost | RandomForest | XGBoost | ...", # TODO: add your methods' names here
    )
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=1)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_stratify", action="store_true", help="Disable stratified split (classification only)")
    p.add_argument(
        "--mnist_max_samples",
        type=int,
        default=12000,
        help="Max MNIST samples (multiclass 0-9); 0 = use full set",
    )
    # p.add_argument("--plot", type=str, default="adaboost_error_curve.png")
    return p.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    mnist_max = 0 if args.mnist_max_samples == 0 else args.mnist_max_samples

    # Load dataset
    X, y, task = load_dataset(
        args.dataset,
        mnist_max_samples=mnist_max,
    )

    # Split dataset into train and test sets
    stratify = task in ("binary", "multiclass") and not args.no_stratify
    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    # Print dataset statistics
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]} | Task: {task}")
    if task == "binary":
        print(f"Class balance (train, y=+1): {np.mean(y_train == 1):.2%}\n")
    elif task == "multiclass":
        print(f"Classes: {np.unique(y_train).size} (labels 0..{np.unique(y_train).size - 1})\n")
    else:
        print(f"Target (train): mean={y_train.mean():.4g}, std={y_train.std():.4g}\n")

    # Main Logic, train and test
    class_num = int(np.max(y)) + 1 if task == "multiclass" else 2
    model = get_model(args.method, task, args, class_num)
    model.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    y_pred = model.predict(X_test)
    if task == "binary":
        y_scores = model.predict_score(X_test)
        acc, auc = evaluate_binary(y_test, y_pred, y_scores)
        print(f"Test Accuracy : {acc:.4f}")
        print(f"Test AUC-ROC  : {auc:.4f}")
    elif task == "multiclass":
        acc = evaluate_multiclass(y_test, y_pred)
        print(f"Test Accuracy : {acc:.4f}")
    else: 
        mse, mae, r2 = evaluate_regression(y_test, y_pred)
        print(f"Test MSE : {mse:.6g}")
        print(f"Test MAE : {mae:.6g}")
        print(f"Test R^2 : {r2:.4f}")
    
    train_err, test_err = get_train_err(model, X_train, y_train, X_test, y_test, task, args.method)
    plot_error_curves(
        train_err,
        test_err,
        title=f"{args.method}: {'MSE' if task == 'regression' else 'Error'} vs. rounds ({args.dataset})",
        outfile=f"{args.method}_{args.dataset}_error_curve.png",
        ylabel="MSE" if task == "regression" else "Error",
    )


if __name__ == "__main__":
    main()
