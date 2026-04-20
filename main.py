#!/usr/bin/env python3
"""
Train an ensemble method on a chosen dataset and report metrics + error curves.
Supports: Single, AdaBoost, Bagging
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    mean_squared_error,
)

from dataset import load_dataset
from ensemble_method import (
    AdaBoostBinaryClassifier,
    AdaBoostMulticlassClassifier,
    AdaBoostRegressor,
)
from utils import (
    evaluate_binary,
    evaluate_multiclass,
    evaluate_regression,
    plot_error_curves,
    get_train_err,
    split_train_test,
)

from gb_methods import (
    GBMachine,
    XGBoostWrapper
)

from base_model import get_base_model
import matplotlib.pyplot as plt


GRADBOOST_RESULTS_DIR = "gradboost_results"
ADABOOST_RESULTS_DIR  = "adaboost_results"
BAGGING_RESULTS_DIR   = "bagging_results"
SINGLE_RESULTS_DIR    = "single_results"

# Methods whose fit() does not take staged X_test/y_test callbacks
INDEPENDENT_FIT_METHODS = {"Bagging", "Single"}


def get_model(args: argparse.Namespace, base_model):
    if args.method == "AdaBoost":
        if args.task == "binary":
            return AdaBoostBinaryClassifier(
                n_estimators=args.n_estimators, max_depth=args.max_depth, base_model=base_model
            )
        elif args.task == "multiclass":
            return AdaBoostMulticlassClassifier(
                n_estimators=args.n_estimators, max_depth=args.max_depth,
                class_num=args.class_num, base_model=base_model,
            )
        elif args.task == "regression":
            return AdaBoostRegressor(
                n_estimators=args.n_estimators, max_depth=args.max_depth, base_model=base_model
            )
        else:
            raise ValueError(f"Unknown task type: {args.task}")

    elif args.method == "Bagging":
        is_clf = args.task in ("binary", "multiclass")
        EnsembleCls = BaggingClassifier if is_clf else BaggingRegressor
        return EnsembleCls(
            estimator=base_model,
            n_estimators=args.n_estimators,
            max_samples=1.0,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=args.seed,
        )

    elif args.method == "Single":
        return base_model

    elif args.method == "GradBoost":
        loss = "lad" if args.task in ("binary", "multiclass") else "ls"
        return GBMachine(
            base_model=base_model,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            loss=loss,
            task=args.task,
        )

    elif args.method == "XGBoost":
        loss = "lad" if args.task in ("binary", "multiclass") else "ls"
        return XGBoostWrapper(
            base_model=base_model,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            loss=loss,
            task=args.task,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")


def parse_args():
    p = argparse.ArgumentParser(description="Ensemble methods on tabular / MNIST data")
    p.add_argument("--dataset", type=str, default="adult",
                   help="adult | communities_crime | mnist | ca_housing | allstate | sberbank")
    p.add_argument("--method", type=str, default="AdaBoost",
                   help="Single | AdaBoost | Bagging | GradBoost | XGBoost")
    p.add_argument("--base_model", type=str, default="DecisionTree",
                   help="DecisionTree | SVM | Ridge | LR | NB | MNB")
    p.add_argument("--kernel", type=str, default="rbf")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Ridge regularization / MNB smoothing")
    p.add_argument("--var_smoothing", type=float, default=1e-9)
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_stratify", action="store_true")
    p.add_argument("--mnist_max_samples", type=int, default=12000,
                   help="Max MNIST samples; 0 = full set")
    return p.parse_args()


def _base_model_tag(args) -> str:
    name = args.base_model
    if args.base_model == "DecisionTree":
        return f"{name}_depth_{args.max_depth}"
    if args.base_model in ("Ridge", "MNB"):
        return f"{name}_alpha_{args.alpha}"
    if args.base_model == "LR":
        return f"{name}_C_{args.C}"
    if args.base_model == "NB":
        return f"{name}_var_smoothing_{args.var_smoothing}"
    return name


def results_dir(method: str) -> str:
    return {
        "Bagging": BAGGING_RESULTS_DIR,
        "AdaBoost": ADABOOST_RESULTS_DIR,
        "GradBoost": GRADBOOST_RESULTS_DIR,
        "XGBoost": GRADBOOST_RESULTS_DIR,
    }.get(method, SINGLE_RESULTS_DIR)


def save_row(results_dir, filename, metrics: dict):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    pd.DataFrame([metrics]).to_csv(path, index=False)
    print(f"Results saved -> {path}")


def main():
    args = parse_args()
    mnist_max = 0 if args.mnist_max_samples == 0 else args.mnist_max_samples
    tag = _base_model_tag(args)

    X, y, task = load_dataset(args.dataset, mnist_max_samples=mnist_max)
    args.task = task

    stratify = task in ("binary", "multiclass") and not args.no_stratify
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify,
    )

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]} | Task: {task}")
    if task == "binary":
        print(f"Class balance (train, y=+1): {np.mean(y_train == 1):.2%}\n")
    elif task == "multiclass":
        print(f"Classes: {np.unique(y_train).size} (labels 0..{np.unique(y_train).size - 1})\n")
    else:
        print(f"Target (train): mean={y_train.mean():.4g}, std={y_train.std():.4g}\n")

    args.class_num = int(np.max(y)) + 1 if task == "multiclass" else 2

    base_model = get_base_model(args)
    model = get_model(args, base_model)

    if args.method in INDEPENDENT_FIT_METHODS:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, X_test=X_test, y_test=y_test)

    y_pred = model.predict(X_test)

    os.makedirs("./log", exist_ok=True)
    os.makedirs("./fig", exist_ok=True)
    log_path = f"./log/{args.method}_{args.dataset}_{tag}.log"
    outf = open(log_path, "w")

    if task == "binary":
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        else:
            y_scores = model.predict_score(X_test)
        acc, auc = evaluate_binary(y_test, y_pred, y_scores)
        print(f"Test Accuracy : {acc:.4f}")
        print(f"Test AUC-ROC  : {auc:.4f}")
        outf.write(f"Test Accuracy : {acc:.4f}\n")
        outf.write(f"Test AUC-ROC  : {auc:.4f}\n")

        train_acc = accuracy_score(y_train, model.predict(X_train))
        save_row(results_dir(args.method), f"{args.dataset}_{tag}.csv", dict(
            learner=args.method, train_err=1 - train_acc, test_err=1 - acc,
            gen_gap=(1 - acc) - (1 - train_acc), AUC=auc,
            OOB_acc=getattr(model, "oob_score_", None),
        ))

    elif task == "multiclass":
        acc, cm = evaluate_multiclass(y_test, y_pred)
        print(f"Test Accuracy : {acc:.4f}")
        outf.write(f"Test Accuracy : {acc:.4f}\n")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.savefig(f"./fig/{args.method}_{args.dataset}_{tag}_confusion_matrix.png")
        plt.close()

        train_acc = accuracy_score(y_train, model.predict(X_train))
        save_row(results_dir(args.method), f"{args.dataset}_{tag}.csv", dict(
            learner=args.method, train_err=1 - train_acc, test_err=1 - acc,
            gen_gap=(1 - acc) - (1 - train_acc), AUC=None,
            OOB_acc=getattr(model, "oob_score_", None),
        ))

    else:
        mse, mae, r2 = evaluate_regression(y_test, y_pred)
        print(f"Test MSE : {mse:.6g}")
        print(f"Test MAE : {mae:.6g}")
        print(f"Test R^2 : {r2:.4f}")
        outf.write(f"Test MSE : {mse:.6g}\n")
        outf.write(f"Test MAE : {mae:.6g}\n")
        outf.write(f"Test R^2 : {r2:.4f}\n")

        train_rmse = math.sqrt(mean_squared_error(y_train, model.predict(X_train)))
        test_rmse  = math.sqrt(mse)
        save_row(results_dir(args.method), f"{args.dataset}_{tag}.csv", dict(
            learner=args.method, train_RMSE=train_rmse, test_RMSE=test_rmse,
            gen_gap=test_rmse - train_rmse, R2=r2,
            OOB_R2=getattr(model, "oob_score_", None),
        ))

    outf.close()

    if args.method in ("AdaBoost", "GradBoost", "XGBoost"):
        train_err, test_err = get_train_err(
            model, X_train, y_train, X_test, y_test, task, args.method
        )
        plot_error_curves(
            train_err,
            test_err,
            title=f"{args.method}: {'MSE' if task == 'regression' else 'Error'} vs. rounds ({args.dataset})",
            outfile=f"./fig/{args.method}_{args.dataset}_{tag}_error_curve.png",
            ylabel="MSE" if task == "regression" else "Error",
        )


if __name__ == "__main__":
    main()
