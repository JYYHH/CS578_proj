"""
Weak learners used by ensemble methods.
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def make_base_estimator(max_depth: int = 1) -> DecisionTreeClassifier:
    """Decision tree weak learner for classification (AdaBoost.M1 / SAMME)."""
    return DecisionTreeClassifier(max_depth = max_depth)


def make_base_regressor(max_depth: int = 1) -> DecisionTreeRegressor:
    """Decision tree weak learner for AdaBoost.R2 (regression)."""
    return DecisionTreeRegressor(max_depth = max_depth)
