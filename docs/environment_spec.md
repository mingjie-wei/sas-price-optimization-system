# Baseline environment specification

This document records the modeling choices behind the baseline: which choices are informed by bookings, which are assumptions, and what the simulation leaves out. For the project storyline, start with the [README](../README.md). The implementation is in [environment.py](../hotel_pricing/environment.py), with executable settings in [environment.json](../configs/environment.json).

## What the data suggests

The H2 analysis uses 51,822 reservations arriving in 2015–2016 and reserves 27,508 records from 2017. The split is by arrival cohort, not a strict as-of backtest; some December stays finish after the cutoff. Source, license, and checksum are in [data/source.json](../data/source.json); computed findings are in `outputs/analysis/summary.json`.

| Choice | Baseline setting | Why |
|---|---|---|
| Room type | One type inspired by reserved code A | A is 81.2% of training reservations; the code is anonymous |
| Reference rate | 90 source rate units | Rounded from median ADR 90.95 among 23,210 completed, positive-night, positive-ADR room-A reservations |
| Price menu | 55, 70, 90, 110, 125 | Assumed multipliers 0.6–1.4 around 90, rounded to the nearest 5 |
| Horizon | 330 daily decisions | Room-A p90 lead time is 301 days; adding one day and rounding up gives coverage of lead days 0–329 (93.6%) |
| Capacity | 10 rooms | A toy choice; actual inventory is not in the dataset |
| Stay length | One stay night | A simplification: completed positive-night stays have median length 3; only 20.7% are one night |
| Cancellations | Leave out initially | A meaningful limitation: 40.0% canceled and 1.3% were no-shows |
| Seasonality | One fixed arrival context initially | Only 2016 is a complete training year |

The price scale uses completed positive-night, positive-ADR reservations. Zero-rate and zero-night records remain in the audit. Outliers stay in the statistics; only the histogram display is limited at p99. No currency conversion or inflation adjustment is applied.

## Sensitivity of the horizon

I keep identical-looking records because anonymization can make different reservations look the same. There are 18,513 repeated source-field rows. Removing them changes room-A p90 lead time from **301 to 183 days** and the overall cancellation/no-show rate from **41.3% to 27.7%**. This sensitivity limits how strongly the observed distribution can justify a particular horizon. The baseline retains 330 periods; a shorter horizon would be a separate modeling choice.

## State, action, and reward

- **State:** `(selling_periods_remaining, rooms_remaining)`. With 330 daily decisions, the first period is 329 days before arrival and the last is arrival day. Zero periods means terminal.
- **Action:** choose one of the five prices.
- **Transition:** at most one room sells per period; a sale reduces inventory by one. Time advances whether or not a sale occurs.
- **Reward:** price received for an actual sale; otherwise zero. Gamma is 1 and unsold rooms have zero terminal value.
- **Start/end:** begin with full inventory; end at sellout or horizon expiry. Horizon expiry is terminal for this task, not an external truncation.

Remaining rooms come from the simulator. Guest lead time is descriptive data, not a known attribute of a shopper who has not booked. Final status, assigned room, and cumulative booking changes are excluded from observations because they can reveal later information. Many source fields reflect the day before arrival.

## Demand assumptions

The CSV does not identify shopper volume or conversion. The assumed daily sale model is:

```text
p(price) = sigmoid(logit(p_ref) - beta * (price / reference - 1))
p_ref = 12 / horizon
```

The assumed 12 expected reference-price sales before capacity limits create inventory pressure against 10 rooms. With 330 periods, `p_ref = 0.0364`. This is a combined sale probability, not a fitted shopper-arrival rate. Start with a time-constant response; the analysis does not show that later shoppers are more willing to pay.

The baseline uses beta=4. Alternative demand levels or slopes would be sensitivity scenarios, not estimated elasticities. They are outside the fixed environment used in the current comparison.

## Fixed data for offline learning

The Gymnasium interface is implemented and checked for sale accounting, repeatable seeds, and episode endings. The walkthrough is in [notebook 02](../notebooks/baseline/02_simulator_and_offline_data.ipynb).

The behavior policy chooses prices 55, 70, 90, 110, and 125 with probabilities 10%, 15%, 50%, 15%, and 10% at every state. This is a simple collection policy, not an estimate from bookings. It collects 2,000 full-start episodes using seed 42, retaining no-sale days and terminal transitions. Complete episodes are split 80/10/10 before training. The saved files include episode IDs, action probabilities, seeds, settings, and checksums; the collector refuses to overwrite them.

Training data covers 73.0% of structurally reachable state–price combinations at least once. This leaves gaps despite trying every price overall. One visit is only a basic coverage check, not sufficient statistical support.

The training and evaluation controls are documented in the [baseline comparison record](baseline_comparison.md). Fresh simulator evaluation is separate from learning from fixed data; it does not validate the demand assumptions against real purchasing behavior.
