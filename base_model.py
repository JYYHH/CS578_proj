"""
Weak learners used by ensemble methods.
"""

from sklearn.linear_model import Ridge
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def make_base_estimator(max_depth: int = 1) -> DecisionTreeClassifier:
    """Decision tree weak learner for classification (AdaBoost.M1 / SAMME)."""
    return DecisionTreeClassifier(max_depth=max_depth)


def make_base_regressor(max_depth: int = 1) -> DecisionTreeRegressor:
    """Decision tree weak learner for AdaBoost.R2 (regression)."""
    return DecisionTreeRegressor(max_depth=max_depth)


def make_svc(kernel: str = "rbf", C: float = 1.0) -> SVC:
    return SVC(kernel=kernel, C=C)


def make_svr(kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1) -> SVR:
    return SVR(kernel=kernel, C=C, epsilon=epsilon)


def make_ridge(alpha: float = 1.0) -> Ridge:
    return Ridge(alpha=alpha)
