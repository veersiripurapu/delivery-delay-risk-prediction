# Data Dictionary — Delivery Delay Risk Prediction

Every field in the route dataset, what it means, and — critically — **when it
becomes available**. The "Available when?" column is the heart of the leakage
discipline: anything only known *after completion* is the outcome and must never be
a model input.

## Route identity

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `route_id` | Unique route identifier | RT-000037 | Before dispatch | No (identifier) |
| `route_date` | Date the route runs | 2025-06-14 | Before dispatch | No (used for time split) |
| `dc` | Distribution center | DC_ATLANTA | Before dispatch | Yes (encoded) |
| `region` | Region of the DC | Southeast | Before dispatch | Yes (encoded) |
| `route_type` | Urban / suburban / rural | urban | Before dispatch | Yes (encoded) |
| `vehicle_type` | Box truck / tractor-trailer / van | box_truck | Before dispatch | Yes (encoded) |

## Planned scope

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `planned_stops` | Planned delivery stops | 24 | Before dispatch | Yes |
| `planned_miles` | Planned route distance | 132 | Before dispatch | Yes |
| `planned_cases` | Planned case volume | 780 | Before dispatch | Yes |
| `planned_weight_lbs` | Planned load weight | 20,300 | Before dispatch | Yes |
| `planned_duration_min` | Planned route duration | 510 | Before dispatch | Yes |

## Calendar

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `day_of_week` | Day of the route | Mon | Before dispatch | Yes |
| `is_monday` | Monday flag | 1 | Before dispatch | Yes |
| `is_month_end` | Month-end flag | 0 | Before dispatch | Yes |
| `holiday_week_flag` | Holiday-week flag | 0 | Before dispatch | Yes |

## Warehouse readiness

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `warehouse_load_delay_min` | Minutes loading finished behind plan | 45 | At dispatch | Yes |
| `picker_shortage_flag` | Staffing-shortage flag | 1 | At dispatch | Yes |
| `load_complexity_score` | Load complexity (1–10) | 8 | At dispatch | Yes |

## Dispatch execution

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `start_delay_min` | Minutes actual departure was behind plan | 32 | At departure | Yes |

## Driver

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `driver_tenure_days` | Days the driver has been active | 120 | Before dispatch | Yes |
| `driver_route_familiarity` | Driver has run this route before (1/0) | 1 | Before dispatch | Yes |
| `driver_avg_delay_rate` | Generated dispatch-time driver profile (prior-delay tendency) | 0.22 | Before dispatch | Yes |

> `driver_avg_delay_rate` is a **generated profile attribute**, not a rate computed
> from this dataset's outcomes — so it introduces no leakage. In production it would
> be computed only from routes occurring *before* the route being scored.

## Customer / complexity

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `delivery_window_pressure` | Low / medium / high | high | Before dispatch | Yes (ordinal) |
| `retail_stop_pct` | Share of retail stops | 0.55 | Before dispatch | Yes |
| `restaurant_stop_pct` | Share of restaurant stops | 0.30 | Before dispatch | Yes |

## Environment (forecast / expected)

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `weather_severity` | Forecast weather severity (0–5) | 3 | Before dispatch | Yes |
| `traffic_index` | Expected traffic index (0–100) | 78 | Before dispatch | Yes |

## Outcome columns — NOT model inputs

| Field | Description | Example | Available when | Model input? |
|---|---|---|---|---|
| `actual_duration_min` | Final route duration | 615 | After completion | No — leakage |
| `actual_end_offset_min` | Final minutes vs planned end | 63 | After completion | No — leakage |
| `route_delay_min` | Final minutes behind planned end | 63 | After completion | No — defines the target |
| `is_delayed` | Route finished >30 min late (1/0) | 1 | After completion | Target |

## Engineered features (derived from the fields above, all dispatch-time)

`cases_per_stop`, `miles_per_stop`, `weight_per_case`, `is_new_driver`,
`high_complexity_route`, `load_delay_over_30`, `is_high_traffic`,
`restaurant_heavy`, `complexity_index` (stops × traffic),
`window_pressure_ord`, `start_delay_bucket`.
