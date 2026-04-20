"""
Load experiment results and produce comparison plots.

Usage:
  python plotting.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from main import ADABOOST_RESULTS_DIR, BAGGING_RESULTS_DIR, SINGLE_RESULTS_DIR, GRADBOOST_RESULTS_DIR

PLOTS_DIR = "plots"

DATASETS = ["communities_crime", "ca_housing", "adult", "mnist"]
MODELS   = ["DecisionTree", "Ridge", "MNB", "NB", "LR"]

DATASET_LABELS = {
    "adult":            "Adult Income",
    "mnist":            "MNIST",
    "ca_housing":       "California Housing",
    "communities_crime":"Communities & Crime",
}

METHOD_STYLE = {
    "Single":   dict(color="steelblue",  linestyle="--", marker="o"),
    "AdaBoost": dict(color="darkorange", linestyle="-",  marker="s"),
    "Bagging":  dict(color="seagreen",   linestyle="-",  marker="^"),
}

DEPTH_ORDER = [1, 3, 5, 10, 100000]
DEPTH_LABELS = {100000: "∞"}

ALPHA_ORDER = [0.1, 0.3, 1.0, 3.0, 10.0]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_results(results_dir: str, method: str) -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(results_dir, fname))
        df["method"] = method
        df["config"] = fname.removesuffix(".csv")
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def parse_config(config: str) -> dict:
    for ds in DATASETS:
        if config.startswith(ds + "_"):
            rest = config[len(ds) + 1:]
            for model in MODELS:
                if rest.startswith(model + "_"):
                    param_val = rest[len(model) + 1:]
                    param, value = param_val.rsplit("_", 1)
                    return {"dataset": ds, "model": model, "param": param, "value": float(value)}
    return {}


def load_all() -> pd.DataFrame:
    frames = []
    for d, method in [
        (SINGLE_RESULTS_DIR,   "Single"),
        (ADABOOST_RESULTS_DIR, "AdaBoost"),
        (BAGGING_RESULTS_DIR,  "Bagging"),
        (GRADBOOST_RESULTS_DIR, "GradBoost")
    ]:
        if os.path.isdir(d):
            df = load_results(d, method)
            if not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    parsed = df["config"].apply(parse_config).apply(pd.Series)
    return pd.concat([df, parsed], axis=1)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _metric(dataset: str):
    """Return (column, label, higher_is_better)."""
    if dataset in ("adult", "mnist"):
        return "test_err", "Test Error", False
    return "R2", "Test R²", True


def _savefig(name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")


# ---------------------------------------------------------------------------
# Figure 1: DecisionTree — test metric vs depth, all datasets
# ---------------------------------------------------------------------------

def plot_depth_comparison(df: pd.DataFrame):
    sub = df[df["model"] == "DecisionTree"].copy()
    sub["depth"] = sub["value"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, dataset in zip(axes, ["adult", "mnist", "ca_housing", "communities_crime"]):
        col, ylabel, _ = _metric(dataset)
        dset = sub[sub["dataset"] == dataset]

        for method, style in METHOD_STYLE.items():
            mdf = (dset[dset["method"] == method]
                   .set_index("depth")[col]
                   .reindex(DEPTH_ORDER)
                   .dropna())
            if mdf.empty:
                continue
            ax.plot(range(len(mdf)), mdf.values, label=method, **style)

        ax.set_xticks(range(len(DEPTH_ORDER)))
        ax.set_xticklabels([DEPTH_LABELS.get(d, str(d)) for d in DEPTH_ORDER])
        ax.set_xlabel("Tree depth")
        ax.set_ylabel(ylabel)
        ax.set_title(DATASET_LABELS[dataset])
        ax.legend(fontsize=8)

    fig.suptitle("Single vs AdaBoost vs Bagging — Decision Tree base learner", fontsize=12)
    plt.tight_layout()
    _savefig("depth_comparison.png")


# ---------------------------------------------------------------------------
# Figure 2: Ridge — test R² vs alpha, regression datasets
# ---------------------------------------------------------------------------

def plot_ridge_comparison(df: pd.DataFrame):
    sub = df[df["model"] == "Ridge"].copy()
    reg_datasets = ["ca_housing", "communities_crime"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, dataset in zip(axes, reg_datasets):
        dset = sub[sub["dataset"] == dataset]

        for method, style in METHOD_STYLE.items():
            mdf = (dset[dset["method"] == method]
                   .set_index("value")["R2"]
                   .reindex(ALPHA_ORDER)
                   .dropna())
            if mdf.empty:
                continue
            ax.plot(range(len(mdf)), mdf.values, label=method, **style)

        ax.set_xticks(range(len(ALPHA_ORDER)))
        ax.set_xticklabels([str(a) for a in ALPHA_ORDER])
        ax.set_xlabel("α (regularization)")
        ax.set_ylabel("Test R²")
        ax.set_title(DATASET_LABELS[dataset])
        ax.legend(fontsize=8)

    fig.suptitle("Single vs AdaBoost vs Bagging — Ridge base learner", fontsize=12)
    plt.tight_layout()
    _savefig("ridge_comparison.png")


# ---------------------------------------------------------------------------
# Figure 3: Generalization gap — DecisionTree, all datasets
# ---------------------------------------------------------------------------

def plot_gen_gap(df: pd.DataFrame):
    sub = df[df["model"] == "DecisionTree"].copy()
    sub["depth"] = sub["value"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, dataset in zip(axes, ["adult", "mnist", "ca_housing", "communities_crime"]):
        dset = sub[sub["dataset"] == dataset]

        for method, style in METHOD_STYLE.items():
            mdf = (dset[dset["method"] == method]
                   .set_index("depth")["gen_gap"]
                   .reindex(DEPTH_ORDER)
                   .dropna())
            if mdf.empty:
                continue
            ax.plot(range(len(mdf)), mdf.values, label=method, **style)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(DEPTH_ORDER)))
        ax.set_xticklabels([DEPTH_LABELS.get(d, str(d)) for d in DEPTH_ORDER])
        ax.set_xlabel("Tree depth")
        ax.set_ylabel("Generalization gap")
        ax.set_title(DATASET_LABELS[dataset])
        ax.legend(fontsize=8)

    fig.suptitle("Generalization gap — Decision Tree base learner", fontsize=12)
    plt.tight_layout()
    _savefig("gen_gap_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_all()
    if df.empty:
        print("No results found. Run run_all_experiments.sh first.")
        return

    print(f"Loaded {len(df)} rows across {df['method'].nunique()} methods, "
          f"{df['dataset'].nunique()} datasets.")

    plot_depth_comparison(df)
    plot_ridge_comparison(df)
    plot_gen_gap(df)


if __name__ == "__main__":
    main()
