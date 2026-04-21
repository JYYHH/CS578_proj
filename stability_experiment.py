"""
Empirical LOO instability experiments.

Measures beta-hat = (1/m) sum_i mean_{z in S_test} |L(A(S), z) - L(A(S\\i), z)|

Usage:
  python stability_experiment.py              # run all experiments + plot
  python stability_experiment.py --plot_only  # reload JSON + re-plot
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.parallel import Parallel, delayed

from dataset import (
    load_adult,
    load_ca_housing,
    load_communities_crime,
    load_mnist,
)
from utils import split_train_test

RESULTS_FILE = "stability_results.json"
PLOTS_DIR    = "plots"

DATASET_TITLES = {
    "adult":        "Adult Income",
    "mnist":        "MNIST",
    "ca_housing":   "California Housing",
    "communities":  "Communities & Crime",
}

# Hyperparameter sweeps — must match values already in stability_results.json
SVC_C_VALUES    = [0.01, 0.1, 1.0, 10.0]
RF_N_ESTIMATORS = [10, 50, 100, 200]
RIDGE_ALPHAS    = [0.01, 0.1, 1.0, 10.0, 100.0]
TREE_DEPTHS     = [1, 3, 5, 10, None]


# ---------------------------------------------------------------------------
# IQR helpers
# ---------------------------------------------------------------------------

def iqr(a: np.ndarray):
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


def iqr_errbar(mean, q25, q75):
    return [[max(0.0, mean - q25)], [max(0.0, q75 - mean)]]


# ---------------------------------------------------------------------------
# LOO delta computation
# ---------------------------------------------------------------------------

def loo_delta_clf(i, factory, X_train, y_train, X_test, y_test, loss_full):
    idx = np.concatenate([np.arange(i), np.arange(i + 1, len(X_train))])
    m = factory()
    m.fit(X_train[idx], y_train[idx])
    loss_loo = (m.predict(X_test) != y_test).astype(float)
    return float(np.mean(np.abs(loss_full - loss_loo)))


def loo_delta_reg(i, factory, X_train, y_train, X_test, y_test, pred_full):
    idx = np.concatenate([np.arange(i), np.arange(i + 1, len(X_train))])
    m = factory()
    m.fit(X_train[idx], y_train[idx])
    pred_loo = m.predict(X_test)
    sq_full = (pred_full - y_test) ** 2
    sq_loo  = (pred_loo  - y_test) ** 2
    ab_full = np.abs(pred_full - y_test)
    ab_loo  = np.abs(pred_loo  - y_test)
    return float(np.mean(np.abs(sq_full - sq_loo))), float(np.mean(np.abs(ab_full - ab_loo)))


def measure_stability_clf(factory, X_train, y_train, X_test, y_test,
                          max_loo=None, n_jobs=-1):
    rng = np.random.default_rng(42)
    indices = np.arange(len(X_train))
    if max_loo is not None and max_loo < len(indices):
        indices = rng.choice(indices, size=max_loo, replace=False)

    model_full = factory()
    model_full.fit(X_train, y_train)
    loss_full = (model_full.predict(X_test) != y_test).astype(float)

    deltas = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(loo_delta_clf)(i, factory, X_train, y_train, X_test, y_test, loss_full)
        for i in indices
    )
    deltas = np.array(deltas)
    q25, q75 = iqr(deltas)
    return float(deltas.mean()), q25, q75


def measure_stability_reg(factory, X_train, y_train, X_test, y_test,
                          max_loo=None, n_jobs=-1):
    rng = np.random.default_rng(42)
    indices = np.arange(len(X_train))
    if max_loo is not None and max_loo < len(indices):
        indices = rng.choice(indices, size=max_loo, replace=False)

    model_full = factory()
    model_full.fit(X_train, y_train)
    pred_full = model_full.predict(X_test)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(loo_delta_reg)(i, factory, X_train, y_train, X_test, y_test, pred_full)
        for i in indices
    )
    sq = np.array([r[0] for r in results])
    ab = np.array([r[1] for r in results])
    q25_sq, q75_sq = iqr(sq)
    q25_ab, q75_ab = iqr(ab)
    return float(sq.mean()), q25_sq, q75_sq, float(ab.mean()), q25_ab, q75_ab


# ---------------------------------------------------------------------------
# Experiment runners — save in flat {label: metrics} format
# ---------------------------------------------------------------------------

def run_clf_dataset(name, X_train, y_train, X_test, y_test, max_loo, n_jobs):
    print(f"\n--- {DATASET_TITLES[name]} (classification) ---")
    results = {}

    for depth in TREE_DEPTHS:
        dlabel = "∞" if depth is None else str(depth)
        factory = lambda d=depth: DecisionTreeClassifier(max_depth=d, random_state=42)
        mean, q25, q75 = measure_stability_clf(factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"DTree depth={dlabel}"
        results[label] = {"beta_01": mean, "q25_01": q25, "q75_01": q75}
        print(f"  {label:20s}  beta={mean:.4f}")

    for C in SVC_C_VALUES:
        factory = lambda C=C: LinearSVC(C=C, max_iter=2000)
        mean, q25, q75 = measure_stability_clf(factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"SVC C={C}"
        results[label] = {"beta_01": mean, "q25_01": q25, "q75_01": q75}
        print(f"  {label:20s}  beta={mean:.4f}")

    for n in RF_N_ESTIMATORS:
        factory = lambda n=n: RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=1)
        mean, q25, q75 = measure_stability_clf(factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"RF n={n}"
        results[label] = {"beta_01": mean, "q25_01": q25, "q75_01": q75}
        print(f"  {label:20s}  beta={mean:.4f}")

    for n in RF_N_ESTIMATORS:
        factory = lambda n=n: BaggingClassifier(estimator=LinearSVC(C=1.0, max_iter=2000), n_estimators=n, random_state=42)
        mean, q25, q75 = measure_stability_clf(factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"Bagging(SVC) n={n}"
        results[label] = {"beta_01": mean, "q25_01": q25, "q75_01": q75}
        print(f"  {label:20s}  beta={mean:.4f}")

    return {"task": "binary" if name == "adult" else "multiclass", "results": results}


def run_reg_dataset(name, X_train, y_train, X_test, y_test, max_loo, n_jobs):
    print(f"\n--- {DATASET_TITLES[name]} (regression) ---")
    results = {}

    for depth in TREE_DEPTHS:
        dlabel = "∞" if depth is None else str(depth)
        factory = lambda d=depth: DecisionTreeRegressor(max_depth=d, random_state=42)
        msq, q25sq, q75sq, mab, q25ab, q75ab = measure_stability_reg(
            factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"DTree depth={dlabel}"
        results[label] = {"beta_sq": msq, "q25_sq": q25sq, "q75_sq": q75sq,
                          "beta_abs": mab, "q25_abs": q25ab, "q75_abs": q75ab}
        print(f"  {label:20s}  beta_sq={msq:.4f}  beta_abs={mab:.4f}")

    for alpha in RIDGE_ALPHAS:
        factory = lambda a=alpha: Ridge(alpha=a)
        msq, q25sq, q75sq, mab, q25ab, q75ab = measure_stability_reg(
            factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"Ridge α={alpha}"
        results[label] = {"beta_sq": msq, "q25_sq": q25sq, "q75_sq": q75sq,
                          "beta_abs": mab, "q25_abs": q25ab, "q75_abs": q75ab}
        print(f"  {label:20s}  beta_sq={msq:.4f}  beta_abs={mab:.4f}")

    for n in RF_N_ESTIMATORS:
        factory = lambda n=n: RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=1)
        msq, q25sq, q75sq, mab, q25ab, q75ab = measure_stability_reg(
            factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"RF n={n}"
        results[label] = {"beta_sq": msq, "q25_sq": q25sq, "q75_sq": q75sq,
                          "beta_abs": mab, "q25_abs": q25ab, "q75_abs": q75ab}
        print(f"  {label:20s}  beta_sq={msq:.4f}  beta_abs={mab:.4f}")

    for n in RF_N_ESTIMATORS:
        factory = lambda n=n: BaggingRegressor(estimator=Ridge(), n_estimators=n, random_state=42)
        msq, q25sq, q75sq, mab, q25ab, q75ab = measure_stability_reg(
            factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        label = f"Bagging(Ridge) n={n}"
        results[label] = {"beta_sq": msq, "q25_sq": q25sq, "q75_sq": q75sq,
                          "beta_abs": mab, "q25_abs": q25ab, "q75_abs": q75ab}
        print(f"  {label:20s}  beta_sq={msq:.4f}  beta_abs={mab:.4f}")

    return {"task": "regression", "results": results}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def savefig(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def iqr_errbar(betas, q25s, q75s):
    lo = [max(0.0, b - q) for b, q in zip(betas, q25s)]
    hi = [max(0.0, q - b) for b, q in zip(betas, q75s)]
    return [lo, hi]


def plot_clf_dataset(name: str, results: dict):
    labels = [l for l in results if not l.startswith("RF")]
    betas  = [results[l]["beta_01"] for l in labels]
    q25s   = [results[l]["q25_01"]  for l in labels]
    q75s   = [results[l]["q75_01"]  for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4))
    ax.bar(x, betas, yerr=iqr_errbar(betas, q25s, q75s), capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg. LOO instability beta (0-1 loss)")
    ax.set_title(f"LOO Instability — {DATASET_TITLES[name]}\n(error bars: IQR)")
    plt.tight_layout()
    savefig(f"stability_{name}.png")


def plot_reg_dataset(name: str, results: dict):
    labels    = [l for l in results if not l.startswith("RF") and not l.startswith("Bagging")]
    betas_abs = [results[l]["beta_abs"] for l in labels]
    q25_abs   = [results[l]["q25_abs"]  for l in labels]
    q75_abs   = [results[l]["q75_abs"]  for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), 4))
    ax.bar(x, betas_abs, yerr=iqr_errbar(betas_abs, q25_abs, q75_abs),
           capsize=4, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Avg. LOO instability beta (absolute loss)")
    ax.set_title(f"LOO Instability — {DATASET_TITLES[name]}\n(error bars: IQR)")
    plt.tight_layout()
    savefig(f"stability_{name}.png")


def plot_ensemble_instability(all_results: dict):
    """Line plot: beta vs. n_estimators, one subplot per dataset."""
    dataset_order = ["adult", "mnist", "ca_housing", "communities"]
    datasets = [(n, all_results[n]) for n in dataset_order if n in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.flatten()

    for ax, (name, dataset_res) in zip(axes, datasets):
        results = dataset_res["results"]
        rf_entries = [(int(k.split("=")[1]), v) for k, v in results.items() if k.startswith("RF n=")]
        rf_entries.sort()
        ns = [e[0] for e in rf_entries]

        if dataset_res.get("task") in ("binary", "multiclass"):
            betas = [e[1]["beta_01"] for e in rf_entries]
            ax.plot(ns, betas, marker="o", color="steelblue", label="RF")
            bag_svc_entries = [(int(k.split("=")[1]), v) for k, v in results.items() if k.startswith("Bagging(SVC) n=")]
            if bag_svc_entries:
                bag_svc_entries.sort()
                bag_ns  = [e[0] for e in bag_svc_entries]
                bag_b   = [e[1]["beta_01"] for e in bag_svc_entries]
                ax.plot(bag_ns, bag_b, marker="s", color="darkorange", linestyle="--", label="SVC")
        else:
            betas_abs = [e[1]["beta_abs"] for e in rf_entries]
            ax.plot(ns, betas_abs, marker="o", color="steelblue", label="RF")
            bag_entries = [(int(k.split("=")[1]), v) for k, v in results.items() if k.startswith("Bagging(Ridge) n=")]
            if bag_entries:
                bag_entries.sort()
                bag_ns  = [e[0] for e in bag_entries]
                bag_abs = [e[1]["beta_abs"] for e in bag_entries]
                ax.plot(bag_ns, bag_abs, marker="s", color="darkorange", linestyle="--", label="Ridge")

        ax.legend(fontsize=7)

        ax.set_xlabel("# Estimators")
        ax.set_ylabel(r"Avg. LOO Instability $\hat{\beta}$")
        ax.set_title(DATASET_TITLES.get(name, name))

    fig.suptitle("LOO Instability — Ensemble", fontsize=12)
    plt.tight_layout()
    savefig("stability_rf_vs_estimators.png")


def plot_all(all_results):
    for name, dataset_res in all_results.items():
        task = dataset_res.get("task", "")
        if task in ("binary", "multiclass"):
            plot_clf_dataset(name, dataset_res["results"])
        else:
            plot_reg_dataset(name, dataset_res["results"])
    plot_ensemble_instability(all_results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--n_train", type=int, default=500,
                   help="Training samples to subsample (default 500)")
    p.add_argument("--n_jobs", type=int, default=-1)
    return p.parse_args()


def resample(X, y, n, rng):
    if n >= len(X):
        return X, y
    idx = rng.choice(len(X), size=n, replace=False)
    return X[idx], y[idx]


if __name__ == "__main__":
    args = parse_args()
    rng = np.random.default_rng(42)

    if args.plot_only:
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        plot_all(all_results)
    else:
        all_results = {}

        for name, loader in [("adult", load_adult), ("mnist", lambda: load_mnist(max_samples=5000))]:
            X, y, _ = loader()
            X_train, X_test, y_train, y_test = split_train_test(X, y, stratify=True)
            X_train, y_train = resample(X_train, y_train, args.n_train, rng)
            print(f"  {name}: using {len(X_train)} training samples")
            all_results[name] = run_clf_dataset(name, X_train, y_train, X_test, y_test,
                                                max_loo=None, n_jobs=args.n_jobs)

        for name, loader in [("ca_housing", load_ca_housing),
                              ("communities", load_communities_crime)]:
            X, y, _ = loader()
            X_train, X_test, y_train, y_test = split_train_test(X, y, stratify=False)
            X_train, y_train = resample(X_train, y_train, args.n_train, rng)
            print(f"  {name}: using {len(X_train)} training samples")
            all_results[name] = run_reg_dataset(name, X_train, y_train, X_test, y_test,
                                                max_loo=None, n_jobs=args.n_jobs)

        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {RESULTS_FILE}")

        plot_all(all_results)
