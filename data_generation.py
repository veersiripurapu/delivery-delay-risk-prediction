"""
data_generation.py
===================
Synthetic, leakage-safe route-level logistics data for the
Delivery Delay Risk Prediction project.

Design principles baked in (these are the whole point):

1. DECISION POINT = DISPATCH.
   Every feature in FEATURE_COLUMNS is knowable the moment the truck is
   loaded and leaves. Outcome columns (actual_*, route_delay_min, is_delayed)
   are only known AFTER the route ends and must never be used as features.
   They are written to the CSV on purpose so you practice dropping them.

2. TARGET IS DRIVEN BY A HIDDEN PROPENSITY, NOT A CLEAN FORMULA.
   is_delayed is sampled from a probability built out of the real drivers
   PLUS irreducible noise PLUS unobserved factors (weather event, customer
   backup, equipment issue) that are generated but NEVER exposed as features.
   That noise is what keeps a good model around ~0.80-0.88 AUC instead of a
   fake-looking 0.99.

3. TIME MATTERS.
   Routes span ~12 months so you can do a proper time-based train/test split
   (train on early months, test on later ones) instead of a random split
   that would leak the future into the past.

Ported-to-Databricks note: everything below is plain functions returning a
pandas DataFrame. Swap the final to_csv for spark.createDataFrame(...).write
.saveAsTable("delivery_routes") and the logic is unchanged.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
N_ROUTES = 12_000
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"
DELAY_THRESHOLD_MIN = 30          # a route is "delayed" if it ends >30 min late
RANDOM_SEED = 42

DISTRIBUTION_CENTERS = {
    # dc code -> (region, base traffic level, base delay tendency)
    "DC_ATLANTA":  ("Southeast", 60, 0.05),
    "DC_DALLAS":   ("Southwest", 55, 0.00),
    "DC_CHICAGO":  ("Midwest",   68, 0.10),
    "DC_PHOENIX":  ("Southwest", 48, -0.05),
    "DC_NEWARK":   ("Northeast", 75, 0.15),
    "DC_DENVER":   ("Mountain",  50, 0.00),
}
ROUTE_TYPES = ["urban", "suburban", "rural"]
VEHICLE_TYPES = ["box_truck", "tractor_trailer", "van"]
WINDOW_PRESSURE = ["low", "medium", "high"]

# The columns a model is ALLOWED to see at dispatch time.
# Anything not in this list at training time is either an ID or leakage.
FEATURE_COLUMNS = [
    # planned / scope (known when the route is built)
    "planned_stops", "planned_miles", "planned_cases", "planned_weight_lbs",
    "planned_duration_min",
    # calendar
    "day_of_week", "is_monday", "is_month_end", "holiday_week_flag",
    # warehouse readiness (known when the truck is loaded)
    "warehouse_load_delay_min", "picker_shortage_flag", "load_complexity_score",
    # dispatch execution (known the moment the truck departs)
    "start_delay_min",
    # driver
    "driver_tenure_days", "driver_route_familiarity", "driver_avg_delay_rate",
    # customer / complexity
    "delivery_window_pressure", "retail_stop_pct", "restaurant_stop_pct",
    # environment / context
    "weather_severity", "traffic_index",
    # categorical identity that carries signal
    "dc", "region", "route_type", "vehicle_type",
]

# Columns that are the outcome or only known post-hoc. NEVER features.
LEAKAGE_COLUMNS = [
    "actual_duration_min", "actual_end_offset_min", "route_delay_min",
]
TARGET_COLUMN = "is_delayed"
ID_COLUMNS = ["route_id", "route_date"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def generate(n: int = N_ROUTES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ----- route identity ------------------------------------------------- #
    dc_codes = list(DISTRIBUTION_CENTERS.keys())
    dc = rng.choice(dc_codes, size=n)
    region = np.array([DISTRIBUTION_CENTERS[d][0] for d in dc])
    dc_traffic_base = np.array([DISTRIBUTION_CENTERS[d][1] for d in dc])
    dc_delay_bias = np.array([DISTRIBUTION_CENTERS[d][2] for d in dc])

    route_type = rng.choice(ROUTE_TYPES, size=n, p=[0.45, 0.35, 0.20])
    vehicle_type = rng.choice(VEHICLE_TYPES, size=n, p=[0.55, 0.20, 0.25])

    # dates across the year, business-day weighted
    all_days = pd.bdate_range(START_DATE, END_DATE)
    route_date = pd.Series(pd.to_datetime(rng.choice(all_days.values, size=n)))
    dow = route_date.dt.dayofweek                 # 0 = Monday
    day_of_week = dow.map({0: "Mon", 1: "Tue", 2: "Wed",
                           3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}).to_numpy()
    is_monday = (dow == 0).astype(int).to_numpy()
    is_month_end = (route_date.dt.day >= 25).astype(int).to_numpy()
    holiday_week_flag = rng.binomial(1, 0.06, size=n)  # ~6% of routes

    # ----- planned scope -------------------------------------------------- #
    # rural routes: fewer stops, more miles; urban: more stops, fewer miles
    stops_base = np.where(route_type == "urban", 26,
                  np.where(route_type == "suburban", 20, 12))
    planned_stops = np.clip(
        rng.normal(stops_base, 5).round().astype(int), 4, 45)

    miles_base = np.where(route_type == "rural", 190,
                  np.where(route_type == "suburban", 110, 70))
    planned_miles = np.clip(
        rng.normal(miles_base, 30).round().astype(int), 20, 320)

    planned_cases = np.clip(
        (planned_stops * rng.normal(34, 8, size=n)).round().astype(int), 40, 2000)
    planned_weight_lbs = (planned_cases * rng.normal(26, 4, size=n)).round().astype(int)
    planned_duration_min = np.clip(
        (planned_stops * 14 + planned_miles * 1.4
         + rng.normal(0, 25, size=n)).round().astype(int), 180, 720)

    # ----- warehouse readiness ------------------------------------------- #
    picker_shortage_flag = rng.binomial(1, 0.12, size=n)
    load_complexity_score = np.clip(
        rng.normal(5, 2, size=n) + picker_shortage_flag * 1.5, 1, 10).round(1)
    warehouse_load_delay_min = np.clip(
        rng.gamma(2.0, 9.0, size=n)
        + picker_shortage_flag * 18
        + load_complexity_score * 1.5
        + is_monday * 6, 0, 180).round().astype(int)

    # ----- driver --------------------------------------------------------- #
    driver_tenure_days = np.clip(
        rng.gamma(2.0, 400, size=n), 5, 4000).round().astype(int)
    driver_route_familiarity = rng.binomial(
        1, _sigmoid((driver_tenure_days - 200) / 300)).astype(int)
    driver_avg_delay_rate = np.clip(
        rng.normal(0.22, 0.09, size=n)
        - driver_route_familiarity * 0.04, 0.01, 0.75).round(3)

    # ----- dispatch execution -------------------------------------------- #
    # start delay is partly driven by warehouse delay (cause -> effect),
    # but loosely -- keep them from being near-duplicates of each other
    start_delay_min = np.clip(
        0.35 * warehouse_load_delay_min
        + rng.normal(6, 14, size=n), 0, 150).round().astype(int)

    # ----- customer / complexity ----------------------------------------- #
    retail_stop_pct = np.clip(rng.beta(2, 2, size=n), 0, 1).round(2)
    restaurant_stop_pct = np.clip(1 - retail_stop_pct
                                  - rng.beta(1.5, 6, size=n), 0, 1).round(2)
    window_pressure = rng.choice(WINDOW_PRESSURE, size=n, p=[0.4, 0.4, 0.2])
    window_pressure_num = pd.Series(window_pressure).map(
        {"low": 0, "medium": 1, "high": 2}).to_numpy()

    # ----- environment ---------------------------------------------------- #
    weather_severity = np.clip(
        rng.poisson(1.0, size=n) + holiday_week_flag, 0, 5).astype(int)
    traffic_index = np.clip(
        dc_traffic_base + rng.normal(0, 12, size=n)
        + is_monday * 5
        + np.where(route_type == "urban", 8, 0), 0, 100).round().astype(int)

    # ----- HIDDEN factors (generated, never exposed as features) ---------- #
    hidden_weather_event = rng.binomial(1, 0.05, size=n)      # freak conditions
    hidden_customer_backup = rng.binomial(1, 0.08, size=n)    # dock congestion
    hidden_equipment_issue = rng.binomial(1, 0.04, size=n)    # breakdown

    # ----- delay propensity (the latent logit) ---------------------------- #
    logit = (
        -1.10                                       # base intercept -> ~25% rate
        + 0.70 * _zscore(warehouse_load_delay_min)  # strongest lever, but not sole
        + 0.55 * _zscore(start_delay_min)           # strong
        + 0.60 * _zscore(planned_stops)             # moderate
        + 0.55 * _zscore(traffic_index)             # moderate
        + 0.45 * _zscore(planned_miles) * (route_type == "rural")  # interaction
        + 0.40 * (driver_route_familiarity == 0)    # driver signal
        + 0.45 * _zscore(driver_avg_delay_rate)     # driver signal
        + 0.35 * (window_pressure_num == 2)         # high window pressure
        + 0.25 * is_monday
        + 0.22 * _zscore(load_complexity_score)
        + 0.18 * weather_severity
        + dc_delay_bias                             # DC-level tendency
        # unobserved shocks -- the reason no model can be perfect:
        + 1.4 * hidden_weather_event
        + 1.2 * hidden_customer_backup
        + 1.5 * hidden_equipment_issue
        + rng.normal(0, 1.15, size=n)               # irreducible noise
    )
    prob_delay = _sigmoid(logit)

    # ----- outcomes (LEAKAGE) -------------------------------------------- #
    # minutes late scales with propensity; can be negative (finished early)
    route_delay_min = (
        (prob_delay - 0.5) * 140
        + rng.normal(0, 22, size=n)
    ).round().astype(int)
    is_delayed = (route_delay_min > DELAY_THRESHOLD_MIN).astype(int)
    actual_duration_min = (planned_duration_min + route_delay_min).astype(int)
    actual_end_offset_min = route_delay_min  # alias kept for clarity

    df = pd.DataFrame({
        "route_id": [f"RT-{i:06d}" for i in range(1, n + 1)],
        "route_date": route_date,
        "dc": dc,
        "region": region,
        "route_type": route_type,
        "vehicle_type": vehicle_type,
        "day_of_week": day_of_week,
        "is_monday": is_monday,
        "is_month_end": is_month_end,
        "holiday_week_flag": holiday_week_flag,
        "planned_stops": planned_stops,
        "planned_miles": planned_miles,
        "planned_cases": planned_cases,
        "planned_weight_lbs": planned_weight_lbs,
        "planned_duration_min": planned_duration_min,
        "warehouse_load_delay_min": warehouse_load_delay_min,
        "picker_shortage_flag": picker_shortage_flag,
        "load_complexity_score": load_complexity_score,
        "start_delay_min": start_delay_min,
        "driver_tenure_days": driver_tenure_days,
        "driver_route_familiarity": driver_route_familiarity,
        "driver_avg_delay_rate": driver_avg_delay_rate,
        "delivery_window_pressure": window_pressure,
        "retail_stop_pct": retail_stop_pct,
        "restaurant_stop_pct": restaurant_stop_pct,
        "weather_severity": weather_severity,
        "traffic_index": traffic_index,
        # ---- outcome / leakage columns (drop before training) ----
        "actual_duration_min": actual_duration_min,
        "actual_end_offset_min": actual_end_offset_min,
        "route_delay_min": route_delay_min,
        "is_delayed": is_delayed,
    }).sort_values("route_date").reset_index(drop=True)

    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Chronological split: train on the earliest routes, test on the latest."""
    df = df.sort_values("route_date").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


if __name__ == "__main__":
    df = generate()
    out = "delivery_routes.csv"
    df.to_csv(out, index=False)

    rate = df["is_delayed"].mean()
    print(f"Generated {len(df):,} routes -> {out}")
    print(f"Overall delay rate: {rate:.1%}")
    print(f"Date range: {df.route_date.min().date()} to {df.route_date.max().date()}")

    # A real-sounding business insight, straight from the data:
    lo = df.loc[df.warehouse_load_delay_min <= 15, "is_delayed"].mean()
    hi = df.loc[df.warehouse_load_delay_min > 30, "is_delayed"].mean()
    print(f"\nDelay rate when warehouse load delay <=15 min: {lo:.1%}")
    print(f"Delay rate when warehouse load delay  >30 min: {hi:.1%}")
    print(f"  -> {hi/lo:.1f}x higher delay rate")
