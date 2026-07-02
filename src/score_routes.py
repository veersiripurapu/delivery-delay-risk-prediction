"""
score_routes.py
===============
Load the trained model and score a batch of routes AT DISPATCH TIME, producing a
ranked risk list with Low / Medium / High bands -- the feed for a Power BI
"Delivery Delay Risk Monitor" dashboard.

Key point: this scores routes using ONLY dispatch-time features. It never sees
the outcome columns -- which is the real-world situation the whole project was
designed for. (We even drop them here to prove it.)
"""

from __future__ import annotations
import os
import joblib
import pandas as pd
import data_generation as dg
import preprocessing as pp


def load_model(path: str | None = None):
    """Load the saved model artifact, trying the common locations."""
    candidates = [path] if path else []
    candidates += ["../models/delay_model.joblib", "models/delay_model.joblib",
                   "delay_model.joblib"]
    for c in candidates:
        if c and os.path.exists(c):
            return joblib.load(c)
    raise FileNotFoundError(
        "delay_model.joblib not found. Run 04_model_training.ipynb first.")


def score(df_routes: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Return the routes with a delay_risk probability and risk_band."""
    routes = df_routes.copy()

    # build_feature_matrix expects a target column; at dispatch there is none,
    # so inject a harmless placeholder (features never use the target).
    if dg.TARGET_COLUMN not in routes.columns:
        routes[dg.TARGET_COLUMN] = 0

    X, _ = pp.build_feature_matrix(routes)
    # align to the exact columns the model was trained on
    X = X.reindex(columns=artifact["features"], fill_value=0)

    prob = artifact["model"].predict_proba(X)[:, 1]
    threshold = artifact["threshold"]

    def band(p):
        if p < threshold:
            return "Low"
        if p < 0.50:
            return "Medium"
        return "High"

    keep = ["route_id", "route_date", "dc", "route_type", "planned_stops",
            "planned_miles", "warehouse_load_delay_min", "start_delay_min",
            "driver_route_familiarity"]
    out = df_routes[keep].copy()
    out["delay_risk"] = prob.round(3)
    out["risk_band"] = [band(p) for p in prob]
    out["flagged"] = (prob >= threshold).astype(int)
    return out.sort_values("delay_risk", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    artifact = load_model()

    # Simulate "today's" 300 routes as a dispatcher would see them:
    # dispatch-time view only -- drop every outcome/leakage column.
    today = dg.generate(n=300, seed=999)
    dispatch_view = today.drop(columns=dg.LEAKAGE_COLUMNS + [dg.TARGET_COLUMN])

    ranked = score(dispatch_view, artifact)
    ranked.to_csv("daily_risk_list.csv", index=False)

    print(f"Scored {len(ranked)} routes  (threshold={artifact['threshold']:.2f})")
    print("\nRisk band counts:")
    print(ranked["risk_band"].value_counts().reindex(["High", "Medium", "Low"]))
    print("\nTop 10 highest-risk routes:")
    print(ranked.head(10).to_string(index=False))
    print("\nSaved daily_risk_list.csv")
