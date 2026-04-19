"""
Ensemble learning algorithms (currently AdaBoost with configurable base trees).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostRegressor as sklearn_AdaBoostRegressor
from sklearn.base import clone

class AdaBoostBinaryClassifier:
    """
    AdaBoost.M1 with binary labels in {+1, -1}.

    Weak learners are decision trees from ``make_base_estimator(max_depth)``.
    """

    def __init__(self, base_model, n_estimators: int = 50, max_depth: int = 1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.base_model = base_model
        self.alphas: list = []
        self.learners: list = []
        self.train_errors: list = []
        self.test_errors: list = []

    def fit(self, X, y, X_test = None, y_test = None):
        n = X.shape[0]
        weights = np.ones(n) / n

        for t in range(self.n_estimators):
            stump = clone(self.base_model)
            stump.fit(X, y, sample_weight = weights)

            predictions = stump.predict(X)
            eps = ((predictions * y == -1) * weights).sum()
            eps = np.clip(eps, 1e-10, 1 - 1e-10)
            alpha = 0.5 * np.log((1 - eps) / eps)

            weights = weights * np.exp(-alpha * y * predictions)
            weights /= weights.sum()

            self.learners.append(stump)
            self.alphas.append(alpha)

            train_pred = self.predict(X) # this will cause O(N^2) complexity, since each model is used in every round after training
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
            # print(f"Round {t + 1}/{self.n_estimators} | eps={eps:.4f} | alpha={alpha:.4f} | train_err={self.train_errors[-1]:.4f}")

    def predict_score(self, X):
        scores = np.zeros(X.shape[0])
        for learner, alpha in zip(self.learners, self.alphas):
            scores += alpha * learner.predict(X)
        return scores

    def predict(self, X):
        return np.sign(self.predict_score(X))


class AdaBoostMulticlassClassifier:
    """
    SAMME with multiclass labels {0, 1, 2, ..., K-1}.

    Weak learners are decision trees from ``make_base_estimator(max_depth)``.
    """

    def __init__(self, base_model, n_estimators: int = 50, max_depth: int = 1, class_num: int = 2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_num = class_num
        self.base_model = base_model
        self.alphas: list = []
        self.learners: list = []
        self.train_errors: list = []
        self.test_errors: list = []

    def fit(self, X, y, X_test = None, y_test = None):
        n = X.shape[0]
        weights = np.ones(n) / n

        for t in range(self.n_estimators):
            stump = clone(self.base_model)
            stump.fit(X, y, sample_weight = weights)

            predictions = stump.predict(X)
            eps = ((predictions != y) * weights).sum()
            eps = np.clip(eps, 1e-10, 1 - 1e-10)
            
            # SAMME formula, we only need to do better than random guessing
            alpha = np.log((1 - eps) / eps) + np.log(self.class_num - 1)

            if np.abs(alpha) < 1e-6:
                print(f"  Early stop at round {t + 1}: alpha={alpha:.4f} (weak learner no better than chance)")
                break

            weights = weights * np.exp(alpha * (predictions != y))
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

    def predict_score(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self.class_num))
        row = np.arange(n)
        for learner, alpha in zip(self.learners, self.alphas):
            c = learner.predict(X).astype(int, copy = False)
            scores[row, c] += alpha
        return scores

    def predict(self, X):
        return np.argmax(self.predict_score(X), axis = 1)
    
class AdaBoostRegressor(sklearn_AdaBoostRegressor):
    """AdaBoost.R2 with decision tree regressors (sklearn implementation)."""

    def __init__(self, base_model, n_estimators: int = 50, max_depth: int = 1):
        # sklearn get_params() hard limits 
        self.max_depth = max_depth
        self.base_model = base_model
        '''
            The maximum number of estimators at which boosting is terminated.
            In case of perfect fit, the learning procedure is stopped early.
        '''
        self.n_estimators = n_estimators 
        
        super().__init__(
            estimator = clone(self.base_model),
            n_estimators = n_estimators,
        )
    def fit(self, X, y, X_test = None, y_test = None):
        # just drop the X_test and y_test arguments, since we directly refer the sklearn's implementation
        super().fit(X, y)
