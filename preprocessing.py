"""
preprocessing.py
================
Turns raw route records into model-ready features, and assembles the final
feature matrix. All logic here respects the leakage contract from
data_generation.py: it only ever touches dispatch-time columns.

Two public functions:
  engineer_features(df)     -> df with new engineered columns added
  build_feature_matrix(df)  -> (X, y) ready for scikit-learn

Databricks port note: engineer_features is pure column math (works on a Spark
DataFrame with minor syntax tweaks); build_feature_matrix's get_dummies becomes
a StringIndexer + OneHotEncoder stage. The feature *definitions* stay identical.
"""

from __future__ import annotations
import pandas as pd
import data_generation as dg   # reuse FEATURE_COLUMNS / TARGET_COLUMN / leakage list

# Nominal categoricals -> one-hot encoded. (Unordered: no numeric meaning.)
CATEGORICAL_NOMINAL = ["dc", "region", "route_type", "vehicle_type", "day_of_week"]

# The engineered columns we add on top of the raw features.
ENGINEERED_FEATURES = [
    "cases_per_stop", "miles_per_stop", "weight_per_case",
    "is_new_driver", "high_complexity_route", "load_delay_over_30",
    "is_high_traffic", "restaurant_heavy", "complexity_index",
    "window_pressure_ord", "start_delay_bucket",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Every one is computable at dispatch time."""
    df = df.copy()

    # Intensity ratios -- a 500-case route over 10 stops is very different
    # from 500 cases over 30 stops, even though planned_cases is identical.
    df["cases_per_stop"] = df["planned_cases"] / df["planned_stops"]
    df["miles_per_stop"] = df["planned_miles"] / df["planned_stops"]
    df["weight_per_case"] = df["planned_weight_lbs"] / df["planned_cases"]

    # Business-rule flags the operations team already thinks in.
    df["is_new_driver"] = (df["driver_tenure_days"] < 90).astype(int)
    df["high_complexity_route"] = (
        (df["planned_stops"] > 25) & (df["planned_cases"] > 800)
    ).astype(int)
    df["load_delay_over_30"] = (df["warehouse_load_delay_min"] > 30).astype(int)
    df["is_high_traffic"] = (df["traffic_index"] > 70).astype(int)
    df["restaurant_heavy"] = (df["restaurant_stop_pct"] > 0.5).astype(int)

    # An explicit interaction: stop count scaled by traffic. Tree models can
    # find this themselves, but it makes the linear baseline stronger and the
    # effect easy to explain to stakeholders.
    df["complexity_index"] = df["planned_stops"] * df["traffic_index"] / 100

    # Ordered categoricals -> ordinal codes (order carries meaning).
    df["window_pressure_ord"] = df["delivery_window_pressure"].map(
        {"low": 0, "medium": 1, "high": 2})
    df["start_delay_bucket"] = pd.cut(
        df["start_delay_min"], bins=[-1, 10, 30, 60, 10_000],
        labels=[0, 1, 2, 3]).astype(int)

    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Return (X, y) where X is fully numeric and model-ready.

    - raw numeric features from FEATURE_COLUMNS  (drops the string categoricals)
    - all ENGINEERED_FEATURES
    - one-hot encoded nominal categoricals
    Leakage columns are never included -- X is built only from features.
    """
    df = engineer_features(df)

    raw_numeric = [
        c for c in dg.FEATURE_COLUMNS
        if c not in CATEGORICAL_NOMINAL + ["delivery_window_pressure"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X_numeric = df[raw_numeric + ENGINEERED_FEATURES]
    X_categorical = pd.get_dummies(df[CATEGORICAL_NOMINAL], drop_first=True)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    y = df[dg.TARGET_COLUMN]

    # Safety net: assert no leakage column ever slipped into X.
    leaked = set(X.columns) & set(dg.LEAKAGE_COLUMNS + [dg.TARGET_COLUMN])
    assert not leaked, f"LEAKAGE in feature matrix: {leaked}"

    return X, y


if __name__ == "__main__":
    df = dg.generate()
    X, y = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} columns")
    print(f"Target positive rate: {y.mean():.1%}")
    print("\nColumns:")
    print(list(X.columns))
