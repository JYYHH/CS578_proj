"""
Ensemble learning algorithms (currently AdaBoost with configurable base trees).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from base_model import make_base_estimator


class AdaBoost:
    """
    AdaBoost.M1 with binary labels in {+1, -1}.

    Weak learners are decision trees from ``make_base_estimator(max_depth)``.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alphas: list = []
        self.learners: list = []
        self.train_errors: list = []
        self.test_errors: list = []

    def fit(self, X, y, X_test=None, y_test=None):
        n = X.shape[0]
        weights = np.ones(n) / n

        for t in range(self.n_estimators):
            stump = make_base_estimator(max_depth=self.max_depth)
            stump.fit(X, y, sample_weight=weights)

            predictions = stump.predict(X)
            eps = ((predictions * y == -1) * weights).sum()
            eps = np.clip(eps, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - eps) / eps)

            weights = weights * np.exp(-alpha * y * predictions)
            weights /= weights.sum()

            self.learners.append(stump)
            self.alphas.append(alpha)

            train_pred = self.predict(X)
            self.train_errors.append(1 - accuracy_score(y, train_pred))

            if X_test is not None and y_test is not None:
                test_pred = self.predict(X_test)
                self.test_errors.append(1 - accuracy_score(y_test, test_pred))

            if (t + 1) % 10 == 0:
                print(
                    f"  Round {t + 1}/{self.n_estimators} | "
                    f"eps={eps:.4f} | alpha={alpha:.4f} | "
                    f"train_err={self.train_errors[-1]:.4f}"
                )

    def predict(self, X):
        scores = np.zeros(X.shape[0])
        for learner, alpha in zip(self.learners, self.alphas):
            scores += alpha * learner.predict(X)
        return np.sign(scores)

    def predict_proba(self, X):
        """Real-valued margin before sign (for ROC-AUC)."""
        scores = np.zeros(X.shape[0])
        for learner, alpha in zip(self.learners, self.alphas):
            scores += alpha * learner.predict(X)
        return scores
