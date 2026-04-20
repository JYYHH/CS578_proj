"""
Empirical LOO instability experiments.

Measures beta-hat = (1/m) sum_i mean_{z in S_test} |L(A(S), z) - L(A(S\\i), z)|

Usage:
  python stability_experiment.py            # run all experiments + plot
  python stability_experiment.py --plot_only  # reload JSON + re-plot
  python stability_experiment.py --max_loo 200  # limit LOO iterations (faster)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
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
    "adult":            "Adult Income",
    "mnist":            "MNIST",
    "ca_housing":       "California Housing",
    "communities_crime":"Communities & Crime",
}


# ---------------------------------------------------------------------------
# IQR helpers
# ---------------------------------------------------------------------------

def _iqr(a: np.ndarray):
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


def _iqr_errbar(mean: float, q25: float, q75: float):
    """Asymmetric errbar for matplotlib: [[lower], [upper]], clamped ≥ 0."""
    return [[max(0.0, mean - q25)], [max(0.0, q75 - mean)]]


# ---------------------------------------------------------------------------
# LOO delta computation
# ---------------------------------------------------------------------------

def _loo_delta_clf(i, factory, X_train, y_train, X_test, y_test, loss_full):
    idx = np.concatenate([np.arange(i), np.arange(i + 1, len(X_train))])
    m = factory()
    m.fit(X_train[idx], y_train[idx])
    loss_loo = (m.predict(X_test) != y_test).astype(float)
    return float(np.mean(np.abs(loss_full - loss_loo)))


def _loo_delta_reg(i, factory, X_train, y_train, X_test, y_test, pred_full):
    idx = np.concatenate([np.arange(i), np.arange(i + 1, len(X_train))])
    m = factory()
    m.fit(X_train[idx], y_train[idx])
    pred_loo = m.predict(X_test)
    sq_full  = (pred_full - y_test) ** 2
    sq_loo   = (pred_loo  - y_test) ** 2
    ab_full  = np.abs(pred_full - y_test)
    ab_loo   = np.abs(pred_loo  - y_test)
    return float(np.mean(np.abs(sq_full - sq_loo))), float(np.mean(np.abs(ab_full - ab_loo)))


def measure_stability_clf(factory, X_train, y_train, X_test, y_test,
                          max_loo=None, n_jobs=-1):
    """Returns (mean, q25, q75) of per-sample LOO instability."""
    rng = np.random.default_rng(42)
    indices = np.arange(len(X_train))
    if max_loo is not None and max_loo < len(indices):
        indices = rng.choice(indices, size=max_loo, replace=False)

    model_full = factory()
    model_full.fit(X_train, y_train)
    loss_full = (model_full.predict(X_test) != y_test).astype(float)

    deltas = Parallel(n_jobs=n_jobs)(
        delayed(_loo_delta_clf)(i, factory, X_train, y_train, X_test, y_test, loss_full)
        for i in indices
    )
    deltas = np.array(deltas)
    q25, q75 = _iqr(deltas)
    return float(deltas.mean()), q25, q75


def measure_stability_reg(factory, X_train, y_train, X_test, y_test,
                          max_loo=None, n_jobs=-1):
    """Returns (mean_sq, q25_sq, q75_sq, mean_abs, q25_abs, q75_abs)."""
    rng = np.random.default_rng(42)
    indices = np.arange(len(X_train))
    if max_loo is not None and max_loo < len(indices):
        indices = rng.choice(indices, size=max_loo, replace=False)

    model_full = factory()
    model_full.fit(X_train, y_train)
    pred_full = model_full.predict(X_test)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_loo_delta_reg)(i, factory, X_train, y_train, X_test, y_test, pred_full)
        for i in indices
    )
    sq  = np.array([r[0] for r in results])
    ab  = np.array([r[1] for r in results])
    q25_sq, q75_sq = _iqr(sq)
    q25_ab, q75_ab = _iqr(ab)
    return float(sq.mean()), q25_sq, q75_sq, float(ab.mean()), q25_ab, q75_ab


# ---------------------------------------------------------------------------
# Experiment configs
# ---------------------------------------------------------------------------

SVC_C_VALUES      = [0.01, 0.1, 1.0, 10.0, 100.0]
RF_N_ESTIMATORS   = [1, 5, 10, 25, 50, 100]
TREE_DEPTHS       = [1, 3, 5, 10, None]
RIDGE_ALPHAS      = [0.1, 0.3, 1.0, 3.0, 10.0]


def run_clf_dataset(name, X_train, y_train, X_test, y_test, max_loo, n_jobs):
    print(f"\n--- {DATASET_TITLES[name]} (classification) ---")
    results = {}

    # Decision tree — sweep depth
    dt_means, dt_q25s, dt_q75s = [], [], []
    for depth in TREE_DEPTHS:
        factory = lambda d=depth: DecisionTreeClassifier(max_depth=d, random_state=42)
        mean, q25, q75 = measure_stability_clf(factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        dt_means.append(mean); dt_q25s.append(q25); dt_q75s.append(q75)
        label = f"depth={depth if depth is not None else '∞'}"
        print(f"  DTree {label:12s}  β̂={mean:.4f}")
    results["DTree"] = {"depths": [d if d is not None else 999999 for d in TREE_DEPTHS],
                        "mean": dt_means, "q25": dt_q25s, "q75": dt_q75s}

    # Random forest — sweep n_estimators
    rf_means, rf_q25s, rf_q75s = [], [], []
    for n in RF_N_ESTIMATORS:
        factory = lambda n=n: RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=1)
        mean, q25, q75 = measure_stability_clf(factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        rf_means.append(mean); rf_q25s.append(q25); rf_q75s.append(q75)
        print(f"  RF n={n:4d}               β̂={mean:.4f}")
    results["RF"] = {"n_estimators": RF_N_ESTIMATORS,
                     "mean": rf_means, "q25": rf_q25s, "q75": rf_q75s}

    # SVC linear — sweep C (skip MNIST, too slow)
    if name != "mnist":
        svc_means, svc_q25s, svc_q75s = [], [], []
        for C in SVC_C_VALUES:
            factory = lambda C=C: SVC(kernel="linear", C=C)
            mean, q25, q75 = measure_stability_clf(factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
            svc_means.append(mean); svc_q25s.append(q25); svc_q75s.append(q75)
            print(f"  SVC C={C:<6}           β̂={mean:.4f}")
        results["SVC"] = {"C": SVC_C_VALUES, "mean": svc_means, "q25": svc_q25s, "q75": svc_q75s}

    return results


def run_reg_dataset(name, X_train, y_train, X_test, y_test, max_loo, n_jobs):
    print(f"\n--- {DATASET_TITLES[name]} (regression) ---")
    results = {}

    # Decision tree — sweep depth
    dt_sq, dt_ab = {"mean":[],"q25":[],"q75":[]}, {"mean":[],"q25":[],"q75":[]}
    for depth in TREE_DEPTHS:
        factory = lambda d=depth: DecisionTreeRegressor(max_depth=d, random_state=42)
        msq, q25sq, q75sq, mab, q25ab, q75ab = measure_stability_reg(
            factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        dt_sq["mean"].append(msq); dt_sq["q25"].append(q25sq); dt_sq["q75"].append(q75sq)
        dt_ab["mean"].append(mab); dt_ab["q25"].append(q25ab); dt_ab["q75"].append(q75ab)
        label = f"depth={depth if depth is not None else '∞'}"
        print(f"  DTree {label:12s}  β̂_sq={msq:.4f}  β̂_abs={mab:.4f}")
    results["DTree"] = {"depths": [d if d is not None else 999999 for d in TREE_DEPTHS],
                        "sq": dt_sq, "abs": dt_ab}

    # Random forest — sweep n_estimators
    rf_sq, rf_ab = {"mean":[],"q25":[],"q75":[]}, {"mean":[],"q25":[],"q75":[]}
    for n in RF_N_ESTIMATORS:
        factory = lambda n=n: RandomForestRegressor(n_estimators=n, random_state=42, n_jobs=1)
        msq, q25sq, q75sq, mab, q25ab, q75ab = measure_stability_reg(
            factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        rf_sq["mean"].append(msq); rf_sq["q25"].append(q25sq); rf_sq["q75"].append(q75sq)
        rf_ab["mean"].append(mab); rf_ab["q25"].append(q25ab); rf_ab["q75"].append(q75ab)
        print(f"  RF n={n:4d}               β̂_sq={msq:.4f}  β̂_abs={mab:.4f}")
    results["RF"] = {"n_estimators": RF_N_ESTIMATORS, "sq": rf_sq, "abs": rf_ab}

    # Ridge — sweep alpha
    ridge_sq, ridge_ab = {"mean":[],"q25":[],"q75":[]}, {"mean":[],"q25":[],"q75":[]}
    for alpha in RIDGE_ALPHAS:
        factory = lambda a=alpha: Ridge(alpha=a)
        msq, q25sq, q75sq, mab, q25ab, q75ab = measure_stability_reg(
            factory, X_train, y_train, X_test, y_test, max_loo, n_jobs)
        ridge_sq["mean"].append(msq); ridge_sq["q25"].append(q25sq); ridge_sq["q75"].append(q75sq)
        ridge_ab["mean"].append(mab); ridge_ab["q25"].append(q25ab); ridge_ab["q75"].append(q75ab)
        print(f"  Ridge α={alpha:<6}         β̂_sq={msq:.4f}  β̂_abs={mab:.4f}")
    results["Ridge"] = {"alphas": RIDGE_ALPHAS, "sq": ridge_sq, "abs": ridge_ab}

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _savefig(name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


def plot_clf(name, res):
    fig, axes = plt.subplots(1, len(res), figsize=(5 * len(res), 4))
    if len(res) == 1:
        axes = [axes]

    learner_labels = {"DTree": "Decision Tree (depth)", "RF": "Random Forest (n)", "SVC": "SVC linear (C)"}
    x_vals = {"DTree": [str(d) if d != 999999 else "∞" for d in res["DTree"]["depths"]],
               "RF":   [str(n) for n in RF_N_ESTIMATORS]}
    if "SVC" in res:
        x_vals["SVC"] = [str(c) for c in SVC_C_VALUES]

    for ax, (learner, data) in zip(axes, res.items()):
        xs = range(len(x_vals[learner]))
        means = data["mean"]
        errs  = [_iqr_errbar(m, q25, q75) for m, q25, q75 in zip(means, data["q25"], data["q75"])]
        lowers = [e[0][0] for e in errs]
        uppers = [e[1][0] for e in errs]
        ax.errorbar(xs, means, yerr=[lowers, uppers], fmt="-o", capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_vals[learner])
        ax.set_xlabel(learner_labels[learner])
        ax.set_ylabel(r"Avg. LOO Instability $\hat{\beta}$")
        ax.set_title(learner)

    fig.suptitle(f"LOO Instability — {DATASET_TITLES[name]}", fontsize=12)
    plt.tight_layout()
    _savefig(f"stability_{name}_clf.png")


def plot_reg(name, res, loss="sq"):
    fig, axes = plt.subplots(1, len(res), figsize=(5 * len(res), 4))
    if len(res) == 1:
        axes = [axes]

    x_vals = {"DTree": [str(d) if d != 999999 else "∞" for d in res["DTree"]["depths"]],
               "RF":   [str(n) for n in RF_N_ESTIMATORS],
               "Ridge":[str(a) for a in RIDGE_ALPHAS]}
    learner_labels = {"DTree": "Decision Tree (depth)", "RF": "Random Forest (n)", "Ridge": "Ridge (α)"}
    ylabel = r"Avg. LOO Instability $\hat{\beta}$" + (" (squared)" if loss == "sq" else " (absolute)")

    for ax, (learner, data) in zip(axes, res.items()):
        d = data[loss]
        xs = range(len(x_vals[learner]))
        means = d["mean"]
        errs  = [_iqr_errbar(m, q25, q75) for m, q25, q75 in zip(means, d["q25"], d["q75"])]
        lowers = [e[0][0] for e in errs]
        uppers = [e[1][0] for e in errs]
        ax.errorbar(xs, means, yerr=[lowers, uppers], fmt="-o", capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(x_vals[learner])
        ax.set_xlabel(learner_labels[learner])
        ax.set_ylabel(ylabel)
        ax.set_title(learner)

    fig.suptitle(f"LOO Instability — {DATASET_TITLES[name]}", fontsize=12)
    plt.tight_layout()
    _savefig(f"stability_{name}_reg_{loss}.png")


def plot_all(all_results):
    for name, res in all_results.items():
        if name in ("adult", "mnist"):
            plot_clf(name, res)
        else:
            plot_reg(name, res, loss="sq")
            plot_reg(name, res, loss="abs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plot_only", action="store_true")
    p.add_argument("--max_loo", type=int, default=None,
                   help="Max LOO iterations per dataset (None = full)")
    p.add_argument("--n_jobs", type=int, default=-1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.plot_only:
        with open(RESULTS_FILE) as f:
            all_results = json.load(f)
        plot_all(all_results)
    else:
        all_results = {}

        # Classification datasets
        for name, loader in [("adult", load_adult), ("mnist", lambda: load_mnist(max_samples=5000))]:
            X, y, _ = loader()
            X_train, X_test, y_train, y_test = split_train_test(X, y, stratify=True)
            all_results[name] = run_clf_dataset(name, X_train, y_train, X_test, y_test,
                                                args.max_loo, args.n_jobs)

        # Regression datasets
        for name, loader in [("ca_housing", load_ca_housing), ("communities_crime", load_communities_crime)]:
            X, y, _ = loader()
            X_train, X_test, y_train, y_test = split_train_test(X, y, stratify=False)
            all_results[name] = run_reg_dataset(name, X_train, y_train, X_test, y_test,
                                                args.max_loo, args.n_jobs)

        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved → {RESULTS_FILE}")

        plot_all(all_results)
