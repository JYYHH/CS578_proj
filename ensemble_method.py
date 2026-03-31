import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import urllib.request

# ============================================================
# 1. Download and Load the Dataset
# ============================================================

def load_adult_dataset():
    """
    Downloads and loads the UCI Adult Income dataset.
    Returns X (features) and y (labels) as numpy arrays.
    Labels: +1 for income >50K, -1 for income <=50K.
    """
    url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    url_test  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    col_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    print("Downloading dataset...")
    train_df = pd.read_csv(url_train, header=None, names=col_names,
                           na_values=" ?", skipinitialspace=True)
    test_df  = pd.read_csv(url_test,  header=None, names=col_names,
                           na_values=" ?", skipinitialspace=True, skiprows=1)

    df = pd.concat([train_df, test_df], ignore_index=True)
    return df


# ============================================================
# 2. Preprocessing
# ============================================================

def preprocess(df):
    """
    Handles missing values and encodes categorical features.
    Returns X (numpy array) and y (numpy array of +1/-1 labels).
    """
    # Drop rows with missing values
    df = df.dropna()

    # Clean income column (test set has trailing periods)
    df["income"] = df["income"].str.strip().str.rstrip(".")

    # Encode labels as +1 / -1 (AdaBoost convention)
    df["label"] = df["income"].apply(lambda x: 1 if x == ">50K" else -1)

    # Separate features and label
    X_df = df.drop(columns=["income", "label"])
    y    = df["label"].values

    # Label-encode all categorical columns
    for col in X_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    X = X_df.values.astype(float)
    return X, y


# ============================================================
# 3. AdaBoost Implementation
# ============================================================

class AdaBoost:
    """
    AdaBoost classifier using decision stumps (depth-1 trees) as weak learners.

    Algorithm recap (Freund & Schapire, 1997):
        Initialize: w_i = 1/n for all i
        For t = 1, ..., T:
            1. Train weak learner h_t on weighted dataset
            2. Compute weighted error: eps_t = sum(w_i * 1[h_t(x_i) != y_i])
            3. Compute learner weight: alpha_t = 0.5 * log((1 - eps_t) / eps_t)
            4. Update sample weights: w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))
            5. Normalize weights: w_i <- w_i / sum(w)
        Final: H(x) = sign(sum_t alpha_t * h_t(x))
    """

    def __init__(self, n_estimators = 50, max_depth = 1):
        self.n_estimators = n_estimators
        self.alphas       = []       # learner weights alpha_t
        self.learners     = []       # weak learners h_t
        self.train_errors = []       # track training error per round
        self.test_errors  = []       # track test error per round (filled during fit if X_test given)
        self.max_depth    = max_depth

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Fit AdaBoost on training data (X, y) with labels in {+1, -1}.

        Args:
            X        : np.ndarray of shape (n_samples, n_features)
            y        : np.ndarray of shape (n_samples,) with values in {+1, -1}
            X_test   : optional, for tracking test error per round
            y_test   : optional, for tracking test error per round
        """
        n = X.shape[0]

        # Initialize uniform weights
        weights = np.ones(n) / n

        for t in range(self.n_estimators):
            # --- 3.1: Train a weak learner (decision stump) on weighted data ---
            stump = DecisionTreeClassifier(max_depth = self.max_depth)
            stump.fit(X, y, sample_weight = weights)

            # --- 3.2: Get predictions from the weak learner ---
            predictions = stump.predict(X)

            # --- 3.3: Compute the weighted error eps_t ---
            eps = ((predictions * y == -1) * weights).sum()

            # Clip epsilon to avoid log(0)
            eps = np.clip(eps, 1e-10, 1 - 1e-10)

            # --- 3.4: Compute alpha_t ---
            alpha = 0.5 * np.log((1 - eps) / eps)

            # --- 3.5: Update sample weights ---
            # w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))
            weights = weights * np.exp(-alpha * y * predictions)

            # --- 3.6: Normalize weights so they sum to 1 ---
            weights /= weights.sum()

            # Store learner and its weight
            self.learners.append(stump)
            self.alphas.append(alpha)

            # Track training error
            train_pred = self.predict(X)
            self.train_errors.append(1 - accuracy_score(y, train_pred))

            # Optionally track test error
            if X_test is not None and y_test is not None:
                test_pred = self.predict(X_test)
                self.test_errors.append(1 - accuracy_score(y_test, test_pred))

            if (t + 1) % 10 == 0:
                print(f"  Round {t+1}/{self.n_estimators} | "
                      f"eps={eps:.4f} | alpha={alpha:.4f} | "
                      f"train_err={self.train_errors[-1]:.4f}")
    
    def predict(self, X):
        """
        Predict class labels for X.
        Final prediction: H(x) = sign(sum_t alpha_t * h_t(x))
        """
        # --- 3.7: Compute the weighted sum of weak learner predictions ---
        sum_ = np.zeros(X.shape[0])
        for learner, alpha in zip(self.learners, self.alphas):
            prediction = learner.predict(X)
            sum_ += alpha * prediction
        return np.sign(sum_)


    def predict_proba(self, X):
        """
        Returns soft scores (before sign) — useful for AUC computation.
        """
        scores = np.zeros(X.shape[0])
        for learner, alpha in zip(self.learners, self.alphas):
            scores += alpha * learner.predict(X)
        return scores


# ============================================================
# 4. Plotting Utilities
# ============================================================

def plot_error_curves(ada):
    """Plots training (and optionally test) error vs. boosting round."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(ada.train_errors) + 1), ada.train_errors,
             label="Train Error", linewidth=2)
    if ada.test_errors:
        plt.plot(range(1, len(ada.test_errors) + 1), ada.test_errors,
                 label="Test Error", linewidth=2, linestyle="--")
    plt.xlabel("Boosting Round")
    plt.ylabel("Classification Error")
    plt.title("AdaBoost: Error vs. Boosting Rounds (UCI Adult)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("adaboost_error_curve.png", dpi=150)
    plt.show()
    print("Plot saved to adaboost_error_curve.png")


# ============================================================
# 5. Main Script
# ============================================================

if __name__ == "__main__":

    # Load and preprocess
    df       = load_adult_dataset()
    X, y     = preprocess(df)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Class balance (train): {np.mean(y_train == 1):.2%} positive\n")

    # ---- 5.1: Instantiate and train your AdaBoost ----
    # Suggestion: start with n_estimators=100
    ada = AdaBoost(n_estimators = 200, max_depth = 1)
    ada.fit(X_train, y_train)

    # ---- 5.2: Evaluate on test set ----
    # Compute accuracy and AUC-ROC
    # Hint: use ada.predict() for accuracy, ada.predict_proba() for AUC
    y_pred   = ada.predict(X_test)
    y_scores = ada.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test AUC-ROC  : {auc:.4f}")

    # ---- Plot error curves ----
    plot_error_curves(ada)