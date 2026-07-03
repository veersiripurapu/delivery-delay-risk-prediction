# Delivery Delay Risk Prediction

**Predict, at route departure, whether a delivery route will finish late - so operations can intervene before the final delay occurs instead of reporting it afterward.**

An end-to-end machine-learning project: leakage-safe data engineering → feature engineering → modeling with cost-based threshold selection and SHAP explainability → batch scoring → a four-page Power BI operations dashboard.

> **Status:** Complete (v1). Python pipeline + Power BI dashboard. Databricks/MLflow productionization is the planned v2 (see Roadmap).

---
## Architecture

![End-to-end architecture](Screenshots/Delivery%20delay%20risk%20architecture.png)

## Dashboard preview

*Executive Summary - how much risk exists and where it's concentrated.*
![Executive Summary](Screenshots/Delivery%20Delay%20Risk%20-%20Summary.png)

*Model Performance & Governance - how the model performed and what its limits are.*
![Model Performance & Governance](Screenshots/Model%20Performance%20.png)
---

## Results at a glance

Gradient Boosting classifier, evaluated on a chronological hold-out (train on the earliest ~80% of routes, test on the latest ~20% - a 2,400-route test set):

| Metric | Value |
|---|---|
| ROC-AUC | 0.838 |
| Recall (delays caught) | 0.79 |
| Precision | 0.51 |
| F1 | 0.62 |
| Operating threshold | 0.21 (chosen under a 3:1 false-negative cost assumption) |
| Historical test routes flagged | ~40% |

At the naive 0.5 threshold the model catches only ~44% of delays. Retuning the cutoff to reflect an illustrative business assumption — that missing a delayed route is roughly three times more costly than unnecessarily reviewing an on-time route — raises recall to ~79%, catching about 4 of every 5 delays.

On the 2,400-route test set at the selected threshold:

- **493** correct delay warnings
- **473** false alerts
- **133** missed delays
- **1,301** on-time routes correctly left unflagged

The model caught 493 of 626 delayed routes (~79%), and about 51% of its alerts corresponded to routes that actually became delayed.

---

## 1. Business problem

Delivery operations typically discover route delays *after* they occur - a customer calls, a receiving window is missed, a service escalation lands on someone's desk. That is reactive. This project moves the decision earlier: score every route at departure, flag the high-risk ones, and let dispatch, warehouse, and customer service intervene while it still matters.

## 2. Objective

A binary classifier that outputs a **delay-risk probability** for each route, usable at route departure, translated into Low / Medium / High risk bands and a ranked daily watch list for the operations team.

## 3. The decision point (the core design constraint)

The whole project hinges on **when** the model runs. It scores each route at **dispatch confirmation / actual departure** — once warehouse-load completion and actual departure time are known, but before the route is completed. That fixes which data is fair game:

| Availability | Examples | Use |
|---|---|---|
| **Known before departure** | planned stops, miles, cases, driver assignment, route type, forecast weather severity, expected traffic index | valid |
| **Known at departure** | warehouse-load delay, actual start delay | valid for this model |
| **Known during route** | stops completed, current lateness, live traffic | future re-scoring model |
| **Known after completion** | actual end time, final duration, `route_delay_min` | leakage |

`route_delay_min` and `actual_*` are the **outcome**, known only after the route ends. Using them as inputs would let the model "predict" the past — the single most common mistake in applied ML. They are deliberately kept in the dataset so the training code has to drop them; a hard assertion in `preprocessing.build_feature_matrix` fails loudly if any outcome column ever reaches the feature matrix.

Weather severity and traffic index are treated as **forecast / expected** dispatch-time values, not post-hoc actuals. `driver_avg_delay_rate` is a **generated dispatch-time driver profile attribute**, not a rate calculated from the dataset's outcomes — so it introduces no leakage.

## 4. Dataset

Synthetic, logistics-style data modeled after common distribution operations (route + warehouse + driver + delivery-complexity + environment). It is **not** based on any company's real data, which keeps the repo publicly shareable.

The data is generated so it is *learnable but not trivial*: the delay outcome is sampled from a latent propensity built from the drivers, **plus irreducible noise plus unobserved shocks** (freak weather, dock congestion, equipment issues) that are generated but never exposed as features. That noise is why a strong model lands near 0.84 AUC rather than a fake-looking 0.99 — it represents the real-world factors the features genuinely can't see. Dataset: 12,000 routes across 2025, ~25% delay rate.

## 5. Target variable

```
is_delayed = 1  if  route_delay_min > 30      # finished >30 min behind plan
           = 0  otherwise
```

A 30-minute threshold avoids treating minor timing variance as a true operational delay. It is a single config value (`DELAY_THRESHOLD_MIN`) and easy to sensitivity-test.

## 6. Features (all known at or before departure)

Planned scope (stops, miles, cases, weight, duration) - calendar (day of week, month-end, holiday week) - warehouse readiness (load delay, picker-shortage flag, load complexity) - dispatch execution (start delay) - driver (tenure, route familiarity, generated prior-delay-rate profile) - customer complexity (delivery-window pressure, retail/restaurant mix) - environment (forecast weather severity, expected traffic index) - identity (DC, region, route type, vehicle type).

Engineered on top of these: `cases_per_stop`, `miles_per_stop`, `is_new_driver`, `high_complexity_route`, `load_delay_over_30`, `complexity_index` (stops x traffic), and ordinal encodings — producing a leakage-safe 47-column feature matrix.

## 7. Modeling

- **Baseline:** Logistic Regression (interpretable, sanity-checks the direction of effects).
- **Primary:** Gradient Boosting (captures nonlinear interactions — e.g. high stops *and* high traffic *and* a new driver compounding).
- **Split:** chronological, not random — mirrors real deployment (train on history, score tomorrow).

## 8. Evaluation (beyond accuracy)

Accuracy is misleading at 25% prevalence (predicting "never delayed" scores 75%). This project reports precision / recall / F1 / ROC-AUC / confusion matrix, plus:

- **Cost-based threshold** — the flag cutoff is chosen under an illustrative 3:1 cost assumption (a missed delay treated as ~3x the cost of a false alarm), not a default 0.5. The selected 0.21 threshold minimizes estimated relative cost under that assumption.
- **Calibration assessment** — compares predicted probabilities with observed delay frequencies to check whether risk scores are systematically over- or under-confident. (This project evaluates calibration; it does not apply a recalibration method such as isotonic/Platt.)
- **SHAP explanations** — per-route "why is this route risky" for the per-route risk explanation view. SHAP values explain the model's predictions; they do not establish that a feature caused the real-world outcome.

Top model risk drivers in the synthetic test data: warehouse-load delay, start delay, route complexity (stops x traffic), driver history, and expected traffic.

## 9. Risk bands

The operating threshold determines whether a route is flagged. Risk bands add a communication layer for prioritization. Boundaries as implemented in `score_routes.py`:

| Delay-risk score | Risk band |
|---|---|
| Below the operating threshold (0.21) | Low |
| Between the threshold and 0.50 | Medium |
| At or above 0.50 | High |

## 10. Power BI dashboard

A four-page "Delivery Delay Risk Monitor" built on the model's scored output (`reports/`, `data/processed/daily_risk_list.csv`):

1. **Executive Summary** — how much risk exists and where it's concentrated (KPI cards, risk-band mix, high-risk routes by DC).
2. **Route Risk Detail** — the operational watch list: every flagged route ranked by risk, colour-coded by band, with the top risk factors visible.
3. **Delay Risk Drivers** — the conditions associated with higher risk (warehouse delay, driver familiarity, route type).
4. **Model Performance & Governance** — a technical appendix supporting assessment of model discrimination, alert trade-offs, the operating threshold, limitations, and intended use (metric KPIs, a business-labelled confusion matrix, a precision-recall tradeoff, a business-cost-by-threshold curve, and a model card).

The flag rate shown on the Executive page reflects a separate demonstration scoring batch; because that batch has a different feature distribution from the historical test set, its flag rate can differ from the ~40% test-set figure. The two percentages describe different datasets and are not expected to match.

Screenshots of all four pages are in [`Screenshots/`](Screenshots).

## 11. How the project files work together

```
src/data_generation.py
        ↓  creates synthetic historical route data
data/raw/delivery_routes.csv
        ↓
notebooks/02_eda.ipynb                inspects distributions and operational patterns
        ↓
src/preprocessing.py                  removes leakage fields, encodes, engineers features
        ↓
notebooks/03_feature_engineering.ipynb demonstrates and validates the feature matrix
        ↓
src/train_model.py                    trains models, evaluates thresholds
        ↓
notebooks/04_model_training.ipynb     ROC-AUC, confusion matrices, calibration, SHAP,
        ↓                             cost trade-offs -> saves models/delay_model.joblib
src/score_routes.py                   scores new routes into a ranked risk feed
        ↓
data/processed/daily_risk_list.csv
        ↓
src/generate_metrics.py               writes model_metrics / confusion_matrix /
        ↓                             threshold_tradeoff CSVs from real predictions
Power BI                              Executive, Watch List, Risk Drivers, Governance
```

The `.py` files contain reusable processing logic. The `.ipynb` notebooks provide an interactive walkthrough of the same stages, including explanations, checks, tables, and visualizations.

## 12. Repository structure

```
delivery-delay-risk-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── Delivery Delay Risk Monitor.pbix     # the Power BI report
├── data/
│   ├── raw/                             # delivery_routes.csv
│   └── processed/                       # daily_risk_list.csv (scored feed)
├── src/
│   ├── data_generation.py              # leakage-safe synthetic generator
│   ├── preprocessing.py                # feature engineering + leakage guardrail
│   ├── train_model.py                  # training + cost-threshold logic
│   ├── score_routes.py                 # batch scoring -> ranked risk list
│   └── generate_metrics.py             # computes governance CSVs from real predictions
├── notebooks/
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_scoring_and_dashboard.ipynb
├── reports/
│   ├── model_metrics.csv
│   ├── confusion_matrix.csv
│   └── threshold_tradeoff.csv
├── models/                              # delay_model.joblib (generated locally; gitignored)
└── Screenshots/
```

## 13. Quickstart

Run all commands from the repository root directory.

```bash
pip install -r requirements.txt
python src/data_generation.py        # writes data/raw/delivery_routes.csv
jupyter lab
```

Then run the notebooks in order — this is the reliable path and it saves the trained model that the scorer and metrics scripts use:

```
02_eda.ipynb
03_feature_engineering.ipynb
04_model_training.ipynb        # saves models/delay_model.joblib
05_scoring_and_dashboard.ipynb # writes data/processed/daily_risk_list.csv
```

After the model is saved by notebook 04, you can regenerate the governance CSVs directly:

```bash
python src/generate_metrics.py       # writes the three reports/*.csv files
```

`generate_metrics.py` uses the threshold and cost ratio actually saved inside the trained model, so the CSVs always reflect the current cost assumption. No metric is hard-coded — all are computed from the model's real test-set predictions.

The notebooks and modules enforce the leakage contract:

```python
import data_generation as dg
df = dg.generate()
train, test = dg.time_split(df, train_frac=0.8)   # chronological
X_train = train[dg.FEATURE_COLUMNS]               # features only - no actuals
y_train = train[dg.TARGET_COLUMN]
```

## 14. Limitations and responsible use

- The project uses synthetic data. Reported performance reflects patterns intentionally embedded in the generator and is **not** evidence of performance on real delivery operations.
- The identified model risk drivers are partly a function of the synthetic-data design and should not be treated as new empirical business findings.
- The 3:1 false-negative cost ratio is an illustrative decision scenario, not a validated financial estimate.
- SHAP explanations describe model behavior; they do not establish causation.
- Weather and traffic values are simulated dispatch-time estimates, not live external feeds.
- The model scores **route** risk, not employee performance. Driver-related features should not be the sole basis for punitive or employment decisions.
- The model is intended to support dispatcher review, not to make route, customer, or staffing decisions automatically.
- A real implementation would require data-quality validation, stakeholder approval, security controls, drift monitoring, fairness review, retraining rules, and outcome monitoring.

## 15. Roadmap

- **v1 (this repo):** Python + scikit-learn, notebooks, cost-based threshold, SHAP, four-page Power BI dashboard.
- **v2 (Databricks):** Delta feature table -> MLflow experiment tracking + model registry -> scheduled batch scoring -> risk scores written back to a table -> Power BI dashboard with row-level security by DC.
- **v3:** a mid-route "re-score" model that adds live progress features once a route is running; live weather/traffic feeds replacing the simulated conditions.

---

*Synthetic data is used as a safe stand-in for enterprise operational data. The overall architecture could be adapted to real route, warehouse, driver, telematics, weather, and traffic feeds. However, the target definition, feature availability, preprocessing rules, model performance, calibration, risk bands, and operating threshold would all need to be revalidated before any production use.*
