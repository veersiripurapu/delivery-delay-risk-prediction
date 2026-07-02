# Delivery Delay Risk Prediction

Predict, at dispatch time, whether a delivery route will finish late — so operations
can act *before* the delay happens instead of reporting it *after*.

> Status: Phase 1 (business design) complete · synthetic dataset generated and
> validated (12,000 routes, 25% delay rate, baseline model AUC ≈ 0.84 on a
> time-based holdout).

---

## 1. Business problem

Delivery operations typically discover route delays *after* they occur — a customer
calls, a receiving window is missed, a service escalation lands on someone's desk.
That is reactive. This project moves the decision earlier: score every route the
moment the truck is loaded, flag the high-risk ones, and let dispatch, warehouse,
and customer service intervene while it still matters.

## 2. Objective

A binary classifier that outputs a **delay-risk probability** for each route,
usable at the point of dispatch, translated into Low / Medium / High risk bands and
a ranked "watch list" for the operations team.

## 3. The decision point (this is the core design constraint)

The whole project hinges on **when** the model runs. It runs at **dispatch** — the
moment the truck is loaded and about to leave. That fixes which data is fair game:

| Availability | Examples | Role |
|---|---|---|
| **Known at dispatch** | planned stops/miles/cases, warehouse load delay, start delay, driver tenure, traffic index, day of week | ✅ valid features |
| **Known mid-route** | stops completed so far, running lateness, live traffic | ⚠️ v2 "re-score" model only |
| **Known at completion** | actual end time, actual duration, `route_delay_min` | ❌ leakage — never features |

`route_delay_min` and `actual_*` are the **outcome**. Using them as inputs would let
the model "predict" the past — the single most common mistake in applied ML. They
are deliberately kept in the dataset so the training code has to drop them; see
`FEATURE_COLUMNS` in `data_generation.py` for the enforced contract.

## 4. Dataset

Synthetic, logistics-style data modeled after common distribution operations
(route + warehouse + driver + delivery-complexity + environment). It is **not** based
on any company's real data, which keeps the repo publicly shareable.

The data is generated so it is *learnable but not trivial*: the delay outcome is
sampled from a latent propensity built from the real drivers, **plus irreducible
noise plus unobserved shocks** (freak weather, dock congestion, equipment issues)
that are generated but never exposed as features. That is why a strong model lands
near 0.84 AUC rather than a fake-looking 0.99 — the noise represents the real-world
factors the features genuinely can't see.

## 5. Target variable

```
is_delayed = 1  if  route_delay_min > 30      # finished >30 min behind plan
           = 0  otherwise
```

A 30-minute threshold avoids treating minor timing variance as a true operational
delay. It is a single config value (`DELAY_THRESHOLD_MIN`) and easy to sensitivity-test.

## 6. Feature groups (all known at dispatch)

- **Planned scope:** stops, miles, cases, weight, planned duration
- **Calendar:** day of week, is-Monday, month-end, holiday week
- **Warehouse readiness:** load delay minutes, picker-shortage flag, load complexity
- **Dispatch execution:** start delay minutes
- **Driver:** tenure, route familiarity, historical delay rate
- **Customer complexity:** delivery-window pressure, retail/restaurant stop mix
- **Environment:** weather severity, traffic index
- **Identity signal:** DC, region, route type, vehicle type

## 7. Modeling approach

- **Baseline:** Logistic Regression (interpretable, sanity check on direction of effects)
- **Primary:** Gradient Boosting / Random Forest (captures the nonlinear interactions —
  e.g. high stops *and* high traffic *and* a new driver compounding)
- **Split:** chronological, not random — train on the earliest ~80% of routes, test on
  the latest -20%, mirroring real deployment (train on history, score tomorrow)

## 8. Evaluation (beyond accuracy)

Accuracy is misleading at 25% prevalence (predicting "never delayed" scores 75%). This
project reports:

- **Precision / recall / F1 / ROC-AUC / confusion matrix**
- **Cost-based threshold** — the flag cutoff is chosen by the real cost asymmetry
  (a missed delay → customer escalation costs far more than a false alarm → a dispatcher
  double-checks a fine route), not a default 0.5
- **Probability calibration** — a "70% risk" must actually mean 70% before risk bands
  are meaningful; calibration curve + isotonic/Platt if needed
- **SHAP explanations** — per-route "why is this route risky" for the root-cause dashboard

## 9. Stakeholders & KPI impact

Stakeholders: transportation manager, dispatch supervisor, warehouse manager, customer
service, sales leadership, operations analytics.

KPIs touched: on-time delivery %, late-route %, average route delay minutes, planned vs
actual route time, delivery-window misses, warehouse dispatch delay, service escalations.

## 10. Repository structure

```
delivery-delay-risk-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # delivery_routes.csv 
│   └── processed/
├── src/
│   ├── data_generation.py   # leakage-safe synthetic generator  
│   ├── preprocessing.py
│   ├── train_model.py
│   └── score_routes.py
├── notebooks/
│   ├── 01_generate_data.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── reports/
│   ├── model_metrics.md
│   └── business_summary.md
└── dashboards/
    └── powerbi_mockup.png
```

## 11. Quickstart

```bash
pip install -r requirements.txt
python src/data_generation.py        # writes delivery_routes.csv
```

In training code, enforce the leakage contract:

```python
import data_generation as dg
df = dg.generate()
train, test = dg.time_split(df, train_frac=0.8)  
X_train = train[dg.FEATURE_COLUMNS]              
y_train = train[dg.TARGET_COLUMN]
```

## 12. Roadmap

- **v1 (this repo):** Python + scikit-learn, notebooks, cost-based threshold, SHAP,
  Power BI mock dashboard
- **v2 (Databricks):** Delta feature table → MLflow experiment tracking + model registry
  → scheduled batch scoring → risk scores written back to a table → Power BI risk
  dashboard with row-level security by DC
- **v3:** mid-route "re-score" model that adds live progress features once a route is running
