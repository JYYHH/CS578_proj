"""
Weak learners used by ensemble methods.
"""

from sklearn.tree import DecisionTreeClassifier


def make_base_estimator(max_depth: int = 1) -> DecisionTreeClassifier:
    """Return a fresh decision tree weak learner (same hyperparameters as ensemble config)."""
    return DecisionTreeClassifier(max_depth=max_depth)
