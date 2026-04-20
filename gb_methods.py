from __future__ import annotations

import numpy as np
from sklearn.base import clone
import xgboost as xgb

_LOSSES = ("ls", "lad")

class GBMachine:
    def __init__(
        self,
        base_model,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        loss: str = "ls",
        task: str = "binary"
    ):
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.loss = loss
        self.task = task

        # binary / regression: list of M weak learners
        self.learners: list = []
        # multiclass OVR: list of M rounds, each a list of K weak learners
        self.mc_learners: list = []
        self.classes_: np.ndarray = np.array([])

        self.train_errors: list = []
        self.test_errors: list = []
        self.F0: float = 0.0

    def _init_prediction(self, y: np.ndarray) -> float:
        if self.loss == "ls":
            return float(np.mean(y))
        return float(np.median(y))

    def _negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        raw = y.astype(float) - F
        if self.loss == "ls":
            return raw
        if self.loss == "lad":
            return np.sign(raw)
        else:
            raise ValueError("Unimplemented Loss")

    def _calc_error(self, y: np.ndarray, F: np.ndarray) -> float:
        if self.loss == "ls":
            return float(np.mean((y - F) ** 2))
        return float(np.mean(np.abs(y - F)))

    @property
    def _error_label(self) -> str:
        return "mse" if self.loss == "ls" else "mae"

    def _fit_binary(self, X: np.ndarray, y: np.ndarray, X_test, y_test):
        self.F0 = self._init_prediction(y)
        F = np.full(len(y), self.F0)

        for m in range(self.n_estimators):
            r = self._negative_gradient(y, F)

            h = clone(self.base_model)
            h.fit(X, r)
            self.learners.append(h)

            F += self.learning_rate * h.predict(X)

            train_err = float(np.mean(np.sign(F) != y))
            self.train_errors.append(train_err)

            if X_test is not None and y_test is not None:
                self.test_errors.append(float(np.mean(self.predict(X_test) != y_test)))

            if (m + 1) % 10 == 0:
                print(
                    f"  Round {m+1}/{self.n_estimators} | train_err={train_err:.4f}"
                    + (f" | test_err={self.test_errors[-1]:.4f}" if self.test_errors else "")
                )

    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray, X_test, y_test):
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        # F[i, k] = accumulated score for sample i belonging to class k
        F = np.zeros((len(y), K))

        for m in range(self.n_estimators):
            learners_this_round = []

            for k, cls in enumerate(self.classes_):
                #   y_k: +1 for class k, -1 for all others
                y_k = np.where(y == cls, 1.0, -1.0)
                r_k = self._negative_gradient(y_k, F[:, k])

                h_k = clone(self.base_model)
                h_k.fit(X, r_k)
                learners_this_round.append(h_k)

                F[:, k] += self.learning_rate * h_k.predict(X)

            self.mc_learners.append(learners_this_round)

            train_pred = self.classes_[np.argmax(F, axis=1)]
            train_err = float(np.mean(train_pred != y))
            self.train_errors.append(train_err)

            if X_test is not None and y_test is not None:
                self.test_errors.append(float(np.mean(self.predict(X_test) != y_test)))

            if (m + 1) % 10 == 0:
                print(
                    f"  Round {m+1}/{self.n_estimators} | train_err={train_err:.4f}"
                    + (f" | test_err={self.test_errors[-1]:.4f}" if self.test_errors else "")
                )

    def _fit_regression(self, X: np.ndarray, y: np.ndarray, X_test, y_test):
        self.F0 = self._init_prediction(y)
        F = np.full(len(y), self.F0)

        for m in range(self.n_estimators):
            r = self._negative_gradient(y, F)

            h = clone(self.base_model)
            h.fit(X, r)
            self.learners.append(h)

            F += self.learning_rate * h.predict(X)

            train_err = self._calc_error(y, F)
            self.train_errors.append(train_err)

            if X_test is not None and y_test is not None:
                self.test_errors.append(self._calc_error(y_test, self.predict(X_test)))

            if (m + 1) % 10 == 0:
                lbl = self._error_label
                print(
                    f"  Round {m+1}/{self.n_estimators} | "
                    f"train_{lbl}={train_err:.6g}"
                    + (f" | test_{lbl}={self.test_errors[-1]:.6g}" if self.test_errors else "")
                )

    def fit(self, X: np.ndarray, y: np.ndarray, X_test=None, y_test=None):
        if self.task == "binary":
            self._fit_binary(X, y, X_test, y_test)
        elif self.task == "multiclass":
            self._fit_multiclass(X, y, X_test, y_test)
        else:
            self._fit_regression(X, y, X_test, y_test)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        scores = np.full(X.shape[0], self.F0)
        for h in self.learners:
            scores += self.learning_rate * h.predict(X)
        return scores

    def predict_score_multiclass(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        F = np.zeros((N, len(self.classes_)))
        for learners_m in self.mc_learners:
            for k, h_k in enumerate(learners_m):
                F[:, k] += self.learning_rate * h_k.predict(X)
        return F

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.task == "binary":
            return np.sign(self.predict_score(X))
        elif self.task == "multiclass":
            scores = self.predict_score_multiclass(X)
            return self.classes_[np.argmax(scores, axis=1)]
        else:
            return self.predict_score(X)


class XGBoostWrapper:
    def __init__(
        self,
        base_model,          # accepted for interface compatibility; ignored
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        loss: str = "ls",
        task: str = "binary",
    ):
        if loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.loss = loss
        self.task = task

        self.train_errors: list = []
        self.test_errors: list = []
        self.classes_: np.ndarray = np.array([])

        self._model = None
        self._label_to_xgb: dict = {}   # original label → 0-based int
        self._xgb_to_label: dict = {}   # 0-based int → original label

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _objective_and_metric(self) -> tuple[str, str]:
        if self.task == "binary":
            return "binary:logistic", "error"
        if self.task == "multiclass":
            return "multi:softprob", "merror"
        # regression
        if self.loss == "ls":
            return "reg:squarederror", "rmse"
        return "reg:absoluteerror", "mae"

    def _remap_labels(self, y: np.ndarray) -> np.ndarray:
        """Map original labels to consecutive 0-based ints for XGBoost."""
        self.classes_ = np.unique(y)
        self._label_to_xgb = {c: i for i, c in enumerate(self.classes_)}
        self._xgb_to_label = {i: c for c, i in self._label_to_xgb.items()}
        return np.array([self._label_to_xgb[c] for c in y])

    def _restore_labels(self, y_xgb: np.ndarray) -> np.ndarray:
        return np.array([self._xgb_to_label[int(i)] for i in y_xgb])

    def _extract_errors(self, results: dict, metric: str, has_test: bool) -> None:
        train_raw = results["validation_0"][metric]
        if self.task == "regression" and self.loss == "ls":
            # XGBoost reports RMSE; square to get MSE to match GBMachine
            self.train_errors = [v ** 2 for v in train_raw]
        else:
            self.train_errors = list(train_raw)
        if has_test:
            test_raw = results["validation_1"][metric]
            if self.task == "regression" and self.loss == "ls":
                self.test_errors = [v ** 2 for v in test_raw]
            else:
                self.test_errors = list(test_raw)

    def _print_progress(self) -> None:
        for m in range(self.n_estimators):
            if (m + 1) % 10 == 0:
                tr = self.train_errors[m] if m < len(self.train_errors) else float("nan")
                msg = f"  Round {m+1}/{self.n_estimators} | train_err={tr:.4f}"
                if m < len(self.test_errors):
                    msg += f" | test_err={self.test_errors[m]:.4f}"
                print(msg)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, X_test=None, y_test=None):
        objective, metric = self._objective_and_metric()

        if self.task in ("binary", "multiclass"):
            y_xgb = self._remap_labels(y)
            y_test_xgb = (
                np.array([self._label_to_xgb[c] for c in y_test])
                if y_test is not None else None
            )
            kwargs = {}
            if self.task == "multiclass":
                kwargs["num_class"] = len(self.classes_)
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=objective,
                eval_metric=metric,
                random_state=0,
                verbosity=0,
                **kwargs,
            )
        else:
            y_xgb = y.astype(float)
            y_test_xgb = y_test.astype(float) if y_test is not None else None
            self._model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective=objective,
                eval_metric=metric,
                random_state=0,
                verbosity=0,
            )

        has_test = X_test is not None and y_test_xgb is not None
        eval_set = [(X, y_xgb), (X_test, y_test_xgb)] if has_test else [(X, y_xgb)]
        self._model.fit(X, y_xgb, eval_set=eval_set, verbose=False)

        self._extract_errors(self._model.evals_result(), metric, has_test)
        self._print_progress()

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._model.predict(X)
        if self.task in ("binary", "multiclass"):
            return self._restore_labels(raw)
        return raw.astype(float)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Raw probability of the positive class (binary) or raw predictions (regression)."""
        if self.task == "binary":
            return self._model.predict_proba(X)[:, 1]
        return self._model.predict(X).astype(float)

    def predict_score_multiclass(self, X: np.ndarray) -> np.ndarray:
        """Probability matrix of shape (N, K)."""
        return self._model.predict_proba(X)