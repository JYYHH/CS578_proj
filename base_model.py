"""
Weak learners used by ensemble methods.
"""

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import argparse

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

def make_lr(C: float = 1.0) -> LogisticRegression:
    return LogisticRegression(C=C)

def make_nb(var_smoothing: float = 1e-9) -> GaussianNB:
    return GaussianNB(var_smoothing=var_smoothing)

def make_mnb(alpha: float = 1.0) -> MultinomialNB:
    return MultinomialNB(alpha=alpha)

def get_base_model(
    args: argparse.Namespace,
):
    if args.base_model == "DecisionTree":
        if args.task == "regression":
            return make_base_regressor(max_depth=args.max_depth)
        else:
            return make_base_estimator(max_depth=args.max_depth)
    elif args.base_model == "SVM":
        if args.task == "regression":
            return make_svr(kernel=args.kernel, C=args.C, epsilon=args.epsilon)
        else:
            return make_svc(kernel=args.kernel, C=args.C)
    elif args.base_model == "Ridge":
        if args.task == "regression":
            return make_ridge(alpha=args.alpha)
        else:
            raise ValueError(f"Ridge can only be used for regression")
    elif args.base_model == "LR":
        if args.task == "regression":
            raise ValueError(f"LogisticRegression can only be used for classification")
        else:
            return make_lr(C=args.C)
    elif args.base_model == "NB":
        if args.task == "regression":
            raise ValueError(f"GaussianNB can only be used for classification")
        else:
            return make_nb(var_smoothing=args.var_smoothing)
    elif args.base_model == "MNB":
        if args.task == "regression":
            raise ValueError(f"MultinomialNB can only be used for classification")
        else:
            return make_mnb(alpha=args.alpha)
    else:
        raise ValueError(f"Unknown base model: {args.base_model}")