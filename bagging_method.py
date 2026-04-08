import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, log_loss

from dataset import load_adult, load_ca_housing


# ============================================================
# Part 1 – BaggingClassifier on UCI Adult
# ============================================================

def run_bagging_classifier():
    print("\n" + "=" * 60)
    print("Part 1: BaggingClassifier  (UCI Adult Income)")
    print("=" * 60)

    X, y, _ = load_adult()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
    print(f"Positive class rate (train): {y_train.mean():.2%}\n")

    # --- Baselines: single tree at each depth ---
    configs = [
        ("Single stump (depth=1)", DecisionTreeClassifier(max_depth=1,    random_state=0)),
        ("Single tree  (depth=2)", DecisionTreeClassifier(max_depth=2,    random_state=0)),
        ("Single tree  (depth=3)", DecisionTreeClassifier(max_depth=3,    random_state=0)),
        ("Single tree  (depth=∞)", DecisionTreeClassifier(max_depth=None, random_state=0)),
    ]
    for label, tree in configs:
        tree.fit(X_train, y_train)
        acc = accuracy_score(y_test, tree.predict(X_test))
        auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
        print(f"{label} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    # --- BaggingClassifier: compare depth=None, 2, 3 ---
    # Key parameters:
    #   n_estimators   – number of bootstrap samples / base learners
    #   max_samples    – fraction (or count) of training points per bag
    #   max_features   – fraction (or count) of features per bag
    #   bootstrap      – sample with replacement (True = bagging)
    #   oob_score      – estimate generalization with out-of-bag samples
    print()
    bag_configs = [
        ("gini,  depth=None", DecisionTreeClassifier(criterion="gini",     max_depth=None)),
        ("gini,  depth=3",    DecisionTreeClassifier(criterion="gini",     max_depth=3)),
        ("gini,  depth=2",    DecisionTreeClassifier(criterion="gini",     max_depth=2)),
        ("xent,  depth=None", DecisionTreeClassifier(criterion="log_loss", max_depth=None)),
        ("xent,  depth=3",    DecisionTreeClassifier(criterion="log_loss", max_depth=3)),
        ("xent,  depth=2",    DecisionTreeClassifier(criterion="log_loss", max_depth=2)),
    ]
    bag_clfs = {}
    for label, base in bag_configs:
        clf = BaggingClassifier(
            estimator    = base,
            n_estimators = 200,
            max_samples  = 1.0,
            max_features = 1.0,
            bootstrap    = True,
            oob_score    = True,
            n_jobs       = -1,
            random_state = 42,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
        acc   = accuracy_score(y_test, clf.predict(X_test))
        auc   = roc_auc_score(y_test, proba[:, 1])
        xent  = log_loss(y_test, proba)
        print(f"Bagging 200 ({label}) | Acc: {acc:.4f} | AUC: {auc:.4f} "
              f"| LogLoss: {xent:.4f} | OOB: {clf.oob_score_:.4f}")
        bag_clfs[label] = clf

    # --- Plot accuracy vs n_estimators for each depth ---
    n_range = [1, 5, 10, 25, 50, 100, 150, 200]
    plt.figure(figsize=(7, 4))
    for label, base in bag_configs:
        acc_list = []
        for n in n_range:
            clf = BaggingClassifier(
                estimator=base, n_estimators=n, max_samples=1.0,
                bootstrap=True, n_jobs=-1, random_state=42
            )
            clf.fit(X_train, y_train)
            acc_list.append(accuracy_score(y_test, clf.predict(X_test)))
        plt.plot(n_range, acc_list, marker="o", label=f"Bagging ({label})")

    plt.xlabel("Number of estimators")
    plt.ylabel("Test accuracy")
    plt.title("BaggingClassifier: accuracy vs. n_estimators (Adult)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bagging_classifier_curve.png", dpi=150)
    plt.show()
    print("Plot saved to bagging_classifier_curve.png")

    return bag_clfs["gini,  depth=None"]


# ============================================================
# Part 2 – Variance Characterization
# ============================================================

def run_variance_experiment(X_train, y_train, X_test, B=50):
    """
    For each tree depth, train B trees on independent bootstrap samples and
    measure the empirical variance of predictions across those trees.
    Then compare to the variance of the bagging ensemble's averaged prediction.

    Var(h(x)) = (1/B) sum_b (h_b(x) - h_bar(x))^2,  averaged over test points
    """
    print("\n" + "=" * 60)
    print("Part 2: Variance Characterization")
    print("=" * 60)

    depths  = [1, 2, 3, None]
    n_train = X_train.shape[0]
    rng     = np.random.default_rng(42)

    mean_vars      = []   # mean prediction variance across test points, per depth
    ensemble_vars  = []   # variance of the ensemble average, per depth

    for depth in depths:
        preds = []   # shape (B, n_test)
        for _ in range(B):
            idx  = rng.integers(0, n_train, size=n_train)   # bootstrap sample
            tree = DecisionTreeClassifier(max_depth=depth)
            tree.fit(X_train[idx], y_train[idx])
            preds.append(tree.predict_proba(X_test)[:, 1])

        preds = np.array(preds)                      # (B, n_test)
        var_per_point   = preds.var(axis=0)          # variance across B trees per test point
        ensemble_pred   = preds.mean(axis=0)         # bagging average
        ensemble_var    = ((preds - ensemble_pred)**2).mean(axis=0)  # residual variance

        mean_vars.append(var_per_point.mean())
        ensemble_vars.append(ensemble_var.mean())

        label = f"depth={depth}" if depth is not None else "depth=∞"
        print(f"{label:12s} | Single tree var: {var_per_point.mean():.4f} "
              f"| Ensemble var: {ensemble_var.mean():.4f} "
              f"| Reduction: {1 - ensemble_var.mean()/var_per_point.mean():.1%}")

    # --- Plot ---
    labels = [f"depth={d}" if d is not None else "depth=∞" for d in depths]
    x = np.arange(len(depths))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, mean_vars,     width, label="Single tree")
    plt.bar(x + width/2, ensemble_vars, width, label=f"Bagging (B={B})")
    plt.xticks(x, labels)
    plt.ylabel("Mean prediction variance")
    plt.title("Variance reduction from bagging by tree depth (Adult)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bagging_variance.png", dpi=150)
    plt.show()
    print("Plot saved to bagging_variance.png")


# ============================================================
# Part 3 – BaggingRegressor on California Housing
# ============================================================

def run_bagging_regressor():
    print("\n" + "=" * 60)
    print("Part 2: BaggingRegressor  (California Housing)")
    print("=" * 60)

    X, y, _ = load_ca_housing()   # y = median house value (100k USD)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]  mean: {y.mean():.2f}\n")

    # --- Baseline: single shallow tree ---
    tree = DecisionTreeRegressor(max_depth=4, random_state=0)
    tree.fit(X_train, y_train)
    tree_rmse = np.sqrt(mean_squared_error(y_test, tree.predict(X_test)))
    tree_r2   = r2_score(y_test, tree.predict(X_test))
    print(f"Single tree   | RMSE: {tree_rmse:.4f} | R²: {tree_r2:.4f}")

    # --- BaggingRegressor ---
    # Aggregation: averages predictions across all base regressors.
    bag_reg = BaggingRegressor(
        estimator    = DecisionTreeRegressor(max_depth=4),
        n_estimators = 200,
        max_samples  = 1.0,
        max_features = 1.0,
        bootstrap    = True,
        oob_score    = True,
        n_jobs       = -1,
        random_state = 42,
    )
    bag_reg.fit(X_train, y_train)

    bag_rmse = np.sqrt(mean_squared_error(y_test, bag_reg.predict(X_test)))
    bag_r2   = r2_score(y_test, bag_reg.predict(X_test))
    print(f"Bagging (200) | RMSE: {bag_rmse:.4f} | R²: {bag_r2:.4f} "
          f"| OOB R²: {bag_reg.oob_score_:.4f}")

    # --- Plot RMSE vs n_estimators ---
    n_range   = [1, 5, 10, 25, 50, 100, 150, 200]
    rmse_list = []
    for n in n_range:
        reg = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=n, max_samples=1.0, bootstrap=True,
            n_jobs=-1, random_state=42
        )
        reg.fit(X_train, y_train)
        rmse_list.append(np.sqrt(mean_squared_error(y_test, reg.predict(X_test))))

    plt.figure(figsize=(7, 4))
    plt.axhline(tree_rmse, color="gray", linestyle=":", label="Single tree")
    plt.plot(n_range, rmse_list, marker="o", color="tab:orange", label="Bagging")
    plt.xlabel("Number of estimators")
    plt.ylabel("Test RMSE")
    plt.title("BaggingRegressor: RMSE vs. n_estimators (CA Housing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bagging_regressor_curve.png", dpi=150)
    plt.show()
    print("Plot saved to bagging_regressor_curve.png")

    return bag_reg


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # load data once, share across experiments
    X, y, _ = load_adult()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    run_bagging_classifier()
    run_variance_experiment(X_train, y_train, X_test)
    run_bagging_regressor()
