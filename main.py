#!/usr/bin/env python3
"""
Train AdaBoost on a chosen dataset and report metrics + error curves.
"""

from __future__ import annotations

import argparse

import numpy as np

from dataset import load_dataset
from ensemble_method import AdaBoost
from utils import evaluate_classification, plot_error_curves, split_train_test


def parse_args():
    p = argparse.ArgumentParser(description="AdaBoost ensemble on tabular / MNIST data")
    p.add_argument(
        "--dataset",
        type=str,
        default="adult",
        help="adult | communities_crime | mnist | allstate | sberbank",
    )
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=1)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_stratify", action="store_true", help="Disable stratified split")
    p.add_argument(
        "--mnist_max_samples",
        type=int,
        default=12000,
        help="Max samples after filtering to digits 0/1 (MNIST only); 0 = use all",
    )
    p.add_argument("--plot", type=str, default="adaboost_error_curve.png")
    return p.parse_args()


def main():
    args = parse_args()
    mnist_max = None if args.mnist_max_samples == 0 else args.mnist_max_samples

    X, y = load_dataset(
        args.dataset,
        mnist_max_samples=mnist_max,
    )

    X_train, X_test, y_train, y_test = split_train_test(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=not args.no_stratify,
    )

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]} | Class balance (train): {np.mean(y_train == 1):.2%} positive\n")

    model = AdaBoost(n_estimators=args.n_estimators, max_depth=args.max_depth)
    model.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    acc, auc = evaluate_classification(y_test, y_pred, y_scores)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test AUC-ROC  : {auc:.4f}")

    title = f"AdaBoost: Error vs. Rounds ({args.dataset})"
    plot_error_curves(model.train_errors, model.test_errors or None, title=title, outfile=args.plot)


if __name__ == "__main__":
    main()
