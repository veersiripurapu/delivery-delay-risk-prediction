# Model Card — Delivery Delay Risk Prediction

## Model

Gradient Boosting Classifier, version 1.0 (scikit-learn `GradientBoostingClassifier`).

## Intended purpose

Rank delivery routes by the estimated risk of finishing more than 30 minutes
behind plan, so operations can prioritize review and intervention before routes
run.

## Intended users

Transportation managers, dispatchers, warehouse supervisors, and operations
analysts.

## Prediction point

Route departure — once warehouse-load completion and actual departure status are
known, but before the route is completed. The model uses only information available
at that moment.

## Data

12,000 synthetic logistics-style route records spanning 2025, with an approximate
25% delay rate. The data is generated from a latent delay propensity plus
irreducible noise and unobserved shocks, so the learnable signal is realistic
rather than perfectly separable.

## Training and evaluation

Chronological (time-based) split, not random:

- Training: earliest ~80% of routes
- Testing: latest ~20% (a 2,400-route hold-out)

This mirrors real deployment — training on history and scoring later routes.

## Performance (on the 2,400-route test set)

| Metric | Value |
|---|---|
| ROC-AUC | 0.838 |
| Recall | 0.79 |
| Precision | 0.51 |
| F1 | 0.62 |

## Operating threshold

0.21, selected using an illustrative 3:1 false-negative cost assumption (a missed
delay treated as roughly three times as costly as a false alarm). The threshold is
the point that minimizes estimated relative cost under that assumption — not the
default 0.5.

## Correct and incorrect outcomes (at the selected threshold)

| Outcome | Count |
|---|---|
| Correct warnings (flagged, actually delayed) | 493 |
| False alerts (flagged, actually on time) | 473 |
| Missed delays (not flagged, actually delayed) | 133 |
| Correctly unflagged (not flagged, on time) | 1,301 |

The model caught 493 of 626 delayed routes (~79%); ~51% of its alerts corresponded
to routes that actually became delayed.

## Risk bands

| Delay-risk score | Band |
|---|---|
| Below 0.21 | Low |
| 0.21 to below 0.50 | Medium |
| 0.50 and above | High |

## Appropriate use

Route prioritization and human review. The model is a decision-support tool.

## Inappropriate use

- Automatic employment or disciplinary decisions
- Punitive driver evaluation (the model scores route risk, not driver performance)
- Fully automated route cancellation or reassignment
- Production use without validation on real operational data

## Known limitations

- Synthetic data — reported performance reflects patterns embedded in the generator
  and is not evidence of real-world performance
- Weather and traffic are simulated dispatch-time estimates, not live feeds
- The 3:1 cost ratio is an illustrative assumption, not a validated financial figure
- SHAP and feature importance describe model behavior, not proven causation
- Performance may not transfer to real delivery operations

## Monitoring required in production

- Data drift and prediction drift
- Recall and precision over time
- Probability calibration
- Alert volume (share of routes flagged)
- Performance broken out by DC and route type
- Fairness review before any use touching people
