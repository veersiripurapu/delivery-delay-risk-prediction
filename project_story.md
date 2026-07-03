# Project Story

A plain-language walkthrough of the Delivery Delay Risk Prediction project — the
narrative behind the code, written for anyone who wants to understand what was
built and why without reading every notebook.

## The problem

Delivery teams often discover that a route ran late only *after* it finishes — a
customer calls, a receiving window is missed, a service escalation lands on
someone's desk. By then it is too late to do anything about it. Reporting tells
you what already happened; it does not help you act in time.

## The goal

Predict route-delay risk **at dispatch time** — the moment a truck is loaded and
departing — so operations can intervene before the delay actually occurs.

## Why this mattered

This is the difference between descriptive and predictive analytics. Traditional
reporting explains what happened after the fact. This project demonstrates how the
same operational data can support proactive decisions: flagging the routes most
likely to run late while there is still time to reprioritize loading, reassign
stops, or warn a customer.

## What I built

- A synthetic logistics dataset modeled on real distribution operations
- A leakage-safe feature pipeline with a hard guardrail against using future data
- A Gradient Boosting classifier that scores each route's delay probability
- A cost-based alert threshold tied to a business assumption, not a default 0.5
- A batch-scoring process that produces a ranked daily risk list
- A four-page Power BI operations dashboard, including a model-governance page

## Key design decisions

**Prediction point.** The model scores a route at departure, once warehouse-load
completion and actual departure status are known — but before the route completes.
Defining this moment precisely is what makes the project realistic rather than a
toy.

**Target.** A route is "delayed" if it finishes more than 30 minutes behind plan.
The 30-minute buffer avoids treating minor timing variance as a real delay.

**Leakage prevention.** Actual end time, final duration, and route-delay minutes
are the outcome — known only after the route ends. They are structurally excluded
from the features, enforced by an assertion that fails loudly if any outcome column
ever reaches the model. Using them would let the model "predict" the past.

**Time-based testing.** The model trained on the earliest routes of the year and
was tested on the latest ones — mirroring how it would run in production (learn
from history, score tomorrow) instead of a random split that leaks the future into
the past.

**Threshold selection.** The alert cutoff was chosen using an illustrative
assumption that a missed delay is roughly three times as costly as an unnecessary
review of an on-time route, and set to the point that minimizes estimated relative
cost under that assumption.

## Results

On a 2,400-route chronological test set:

- ROC-AUC: 0.84
- Recall (delays caught): 79%
- Precision: 51%
- F1: 62%
- Selected threshold: 0.21

At the default 0.5 cutoff the model catches only ~44% of delays. Retuning to the
cost-based threshold raises recall to ~79% — catching about four of every five
delays — at the cost of reviewing more routes.

## Business output

The model ranks every route by risk and writes the results to a CSV that feeds
Power BI. Operations see a daily watch list, sorted by risk, with the top
contributing factors for each route.

## What operations could do with it

- Prioritize which routes to review first each morning
- Investigate late warehouse loading on high-risk lanes
- Monitor flagged routes more closely during execution
- Notify customers earlier when a delivery is at risk
- Rebalance work or reassign stops where appropriate

## Limitations

The data is synthetic, the cost ratio is an illustrative assumption, the model has
not been deployed to production, and the risk drivers describe model behavior — not
proven real-world causes. (See `model_card.md` for the full list.)

## What I learned

The most valuable work was not the modeling itself. It was defining the prediction
point precisely, preventing target leakage, connecting model metrics to business
cost rather than optimizing accuracy in a vacuum, and delivering the output through
a tool business users already understand. Those are the parts that turn a model
into something an operations team can actually use.
