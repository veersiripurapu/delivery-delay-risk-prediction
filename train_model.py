"""
train_model.py
==============
Trains the delay-risk models and provides the evaluation primitives the
notebook uses (cost-based threshold, metric summary). Kept importable so the
notebook, a batch scoring job, and a future Databricks job all share one source
of truth for how the model is built.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score)
import data_generation as dg
import preprocessing as pp


def get_train_test(df: pd.DataFrame | None = None, train_frac: float = 0.8):
    """Chronological split -> aligned model matrices. Train on early routes,
    test on the latest ones (mirrors real deployment)."""
    if df is None:
        df = dg.generate()
    tr, te = dg.time_split(df, train_frac)
    X_train, y_train = pp.build_feature_matrix(tr)
    X_test, y_test = pp.build_feature_matrix(te)
    # align columns in case a category is absent from one split
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return X_train, X_test, y_train, y_test


def train_baseline(X_train, y_train):
    """Logistic Regression with scaling -- interpretable baseline."""
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000)),
    ])
    return model.fit(X_train, y_train)


def train_gbm(X_train, y_train):
    """Gradient Boosting -- captures nonlinear interactions."""
    return GradientBoostingClassifier(random_state=0).fit(X_train, y_train)


def cost_optimal_threshold(y_true, y_prob, cost_fn: float = 10.0,
                           cost_fp: float = 1.0):
    """
    Choose the probability cutoff that minimizes expected operational cost.

    cost_fn = cost of a MISSED delay (false negative): customer escalation,
              missed window -- expensive.
    cost_fp = cost of a FALSE ALARM (false positive): a dispatcher glances at
              a route that turns out fine -- cheap.

    Returns (best_threshold, costs_array, thresholds_array).
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    y_true = np.asarray(y_true)
    costs = []
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        fn = int(((pred == 0) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        costs.append(fn * cost_fn + fp * cost_fp)
    costs = np.array(costs)
    return thresholds[costs.argmin()], costs, thresholds


def summarize(name, y_true, y_prob, threshold: float = 0.5):
    pred = (y_prob >= threshold).astype(int)
    print(f"== {name}  (threshold={threshold:.2f}) ==")
    print(f"  ROC-AUC:   {roc_auc_score(y_true, y_prob):.3f}")
    print(f"  Precision: {precision_score(y_true, pred):.3f}")
    print(f"  Recall:    {recall_score(y_true, pred):.3f}")
    print(f"  F1:        {f1_score(y_true, pred):.3f}")
    return pred


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test()
    baseline = train_baseline(X_train, y_train)
    gbm = train_gbm(X_train, y_train)

    for name, model in [("Logistic Regression", baseline),
                        ("Gradient Boosting", gbm)]:
        prob = model.predict_proba(X_test)[:, 1]
        summarize(name, y_test, prob)
        print()

    joblib.dump(gbm, "gbm_model.joblib")
    print("Saved gbm_model.joblib")
