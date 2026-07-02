"""
generate_metrics.py
===================
Compute model-performance artifacts for the Power BI "Model Performance &
Governance" page. Everything here is CALCULATED from the final model's test-set
predictions -- no metric is hard-coded.

It uses the threshold and cost ratio ACTUALLY SAVED inside delay_model.joblib,
so the CSVs always reflect whatever cost assumption you last trained with.

Run from the project root (with venv active):
    python src/generate_metrics.py

Writes three files into  reports/:
    model_metrics.csv        one row of headline metrics + metadata
    confusion_matrix.csv     four outcomes with business-friendly labels
    threshold_tradeoff.csv   metrics across a sweep of thresholds
"""

from __future__ import annotations
import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

# resolve project paths from THIS file's location (robust to where you run it)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in os.sys.path:
    os.sys.path.insert(0, HERE)

import data_generation as dg
import preprocessing as pp
import train_model as tm

from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

MODEL_VERSION = "v1.0"


def _find_model():
    for c in [os.path.join(ROOT, "models", "delay_model.joblib"),
              os.path.join(HERE, "models", "delay_model.joblib"),
              "models/delay_model.joblib"]:
        if os.path.exists(c):
            return joblib.load(c), c
    raise FileNotFoundError(
        "delay_model.joblib not found. Run notebook 04 first to train + save it.")


def main():
    artifact, path = _find_model()
    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))
    cost_fn, cost_fp = artifact.get("cost_ratio", (3, 1))

    # Rebuild the SAME chronological test split the model was evaluated on.
    df = dg.generate()
    train_df, test_df = dg.time_split(df, train_frac=0.8)
    X_train, X_test, y_train, y_test = tm.get_train_test(df)

    # ---- real test-set predictions ----
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    auc = roc_auc_score(y_test, prob)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    pct_flagged = float(pred.mean())

    train_period = f"{train_df['route_date'].min().date()} to {train_df['route_date'].max().date()}"
    test_period = f"{test_df['route_date'].min().date()} to {test_df['route_date'].max().date()}"

    # ---------- 1) model_metrics.csv ----------
    metrics = pd.DataFrame([{
        "Model Name": "Gradient Boosting Classifier",
        "ROC-AUC": round(auc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1 Score": round(f1, 3),
        "Selected Threshold": round(threshold, 2),
        "% Routes Flagged": round(pct_flagged, 3),
        "False-Negative Cost Ratio": f"{int(cost_fn)}:{int(cost_fp)}",
        "Model Version": MODEL_VERSION,
        "Training Period": train_period,
        "Test Period": test_period,
        "Generated Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }])

    # ---------- 2) confusion_matrix.csv ----------
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cm = pd.DataFrame([
        {"Outcome": "Correctly left alone", "Actual": "On time",
         "Predicted": "Not flagged", "Count": int(tn)},
        {"Outcome": "Missed delay", "Actual": "Delayed",
         "Predicted": "Not flagged", "Count": int(fn)},
        {"Outcome": "Unnecessary review", "Actual": "On time",
         "Predicted": "Flagged", "Count": int(fp)},
        {"Outcome": "Correct warning", "Actual": "Delayed",
         "Predicted": "Flagged", "Count": int(tp)},
    ])

    # ---------- 3) threshold_tradeoff.csv ----------
    sweep = sorted(set(np.round(np.arange(0.05, 0.96, 0.05), 2)).union({round(threshold, 2)}))
    yt = np.asarray(y_test)
    rows = []
    for t in sweep:
        p = (prob >= t).astype(int)
        f_p = int(((p == 1) & (yt == 0)).sum())
        f_n = int(((p == 0) & (yt == 1)).sum())
        rows.append({
            "Threshold": t,
            "Precision": round(precision_score(yt, p, zero_division=0), 3),
            "Recall": round(recall_score(yt, p, zero_division=0), 3),
            "F1 Score": round(f1_score(yt, p, zero_division=0), 3),
            "% Flagged": round(p.mean(), 3),
            "False Positives": f_p,
            "False Negatives": f_n,
            "Estimated Business Cost": int(f_n * cost_fn + f_p * cost_fp),
            "Is Selected Threshold": (round(t, 2) == round(threshold, 2)),
        })
    tradeoff = pd.DataFrame(rows)

    # ---------- write ----------
    out_dir = os.path.join(ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    metrics.to_csv(os.path.join(out_dir, "model_metrics.csv"), index=False)
    cm.to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=False)
    tradeoff.to_csv(os.path.join(out_dir, "threshold_tradeoff.csv"), index=False)

    print(f"Loaded model: {path}")
    print(f"Threshold in model: {threshold:.2f}  |  cost ratio {int(cost_fn)}:{int(cost_fp)}")
    print(f"\nAUC={auc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
          f"F1={f1:.3f}  Flagged={pct_flagged:.0%}")
    print(f"Confusion: TN={tn} FN={fn} FP={fp} TP={tp}")
    print(f"\nWrote 3 files to {out_dir}/")


if __name__ == "__main__":
    main()
