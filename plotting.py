"""
Load experiment results and produce comparison plots/tables.

Usage:
  python plotting.py
"""

import os
import pandas as pd
from main import ADABOOST_RESULTS_DIR, BAGGING_RESULTS_DIR, SINGLE_RESULTS_DIR, GRADBOOST_RESULTS_DIR


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


def main():
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
        else:
            print(f"[skip] {d}/ not found")

    if not frames:
        print("No results found. Run run_all_experiments.sh first.")
        return

    all_results = pd.concat(frames, ignore_index=True)
    print(all_results.to_string(index=False))


if __name__ == "__main__":
    main()
