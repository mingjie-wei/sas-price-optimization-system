# Hotel Pricing

## Repository structure

```text
├── hotel_pricing/             # Simulator, learning, evaluation, and explanations
├── configs/                   # Environment, baseline, and policy-improvement settings
├── notebooks/
│   ├── baseline/              # Booking analysis, simulation, and initial learning
│   ├── baseline_analysis/     # Policy diagnosis and CQL–DQN comparison
│   └── policy_improvement/    # Policy audit and controlled improvement
├── data/                      # Source metadata, bookings, and simulated experience
├── docs/
│   ├── environment_spec.md    # Observations and simulator assumptions
│   ├── baseline_comparison.md # Baseline controls and findings
│   ├── policy_improvement/    # Report, poster, decision log, figures, and evidence
│   └── provenance/            # Artifact identities and migration records
├── outputs/
│   ├── analysis/             # Booking-analysis tables and figures
│   ├── simulator/            # Simulator figures
│   ├── baseline/             # Baseline models and comparison results
│   └── policy_improvement/   # Coverage/training results and report exports
├── scripts/                   # Shared data and notebook utilities
│   └── policy_improvement/    # Study runners, evidence, and report builders
├── tests/                     # Shared infrastructure checks
│   ├── baseline/              # Simulator, learning, and comparison checks
│   └── policy_improvement/    # Explanation and improvement checks
└── requirements-*.txt         # Python dependencies
```

## Pricing is learned from experience.

This capstone with SAS investigates whether pricing decisions can be learned directly from past decisions and outcomes, rather than forecasting demand before selecting prices. The central question is how a policy should adapt as time and available inventory change.

While partner data undergoes confidentiality checks and anonymization, simulation provides a controlled setting for developing the workflow and investigating its limitations.

## Booking data informs the setting.

Public hotel reservations provide evidence about lead times, rates, cancellations, and stays. They do not reveal every offered price, rejected offer, or inventory level. The data therefore informs the simulator rather than supplying a complete history for learning pricing decisions.

The [booking analysis](notebooks/baseline/01_hotel_booking_patterns.ipynb) establishes this groundwork. The [environment specification](docs/environment_spec.md) separates observed patterns from modeling assumptions.

## Simulation makes decisions testable.

The simulated hotel starts with 330 days and 10 rooms for one arrival date and one night. Each day, a policy chooses a price from the time and rooms remaining. At most one room sells; unsold inventory carries forward. Both sales and no-sales are recorded.

This small environment allows the best achievable policy to be calculated exactly. The [simulator notebook](notebooks/baseline/02_simulator_and_offline_data.ipynb) describes the assumptions and collection of offline training data.

## The baseline reveals limited adaptation.

Conservative Q-Learning (CQL) learns price scores from a fixed dataset. The learned policy improves on the data-collection policy but mostly behaves like a fixed-price rule. The [CQL experiment](notebooks/baseline/03_offline_cql.ipynb) and [end-to-end summary](notebooks/baseline/04_end_to_end_demo.ipynb) establish the initial workflow.

The [policy-gap diagnosis](notebooks/baseline_analysis/01_policy_gap.ipynb) locates weak decisions and compares them with training support. The [CQL–DQN comparison](notebooks/baseline_analysis/02_cql_vs_offline_dqn.ipynb) tests whether removing CQL's conservative penalty helps under matched settings. It does not support that change as a remedy. The [experiment record](docs/baseline_comparison.md) retains the controls and findings.

## Explanations guide controlled improvement.

Policy maps, score-response curves, and Shapley explanations reveal limited adaptation across time and inventory. The [policy audit](notebooks/policy_improvement/01_policy_audit.ipynb) examines these weaknesses and the first unsuccessful attempt to add complete selling periods.

The [learning-improvement notebook](notebooks/policy_improvement/02_learning_improvement.ipynb) then checks learning mechanics, tests replay and targeted observations, and confirms a fixed configuration on fresh data and training seeds. The revised model improves full-period revenue, with most improvement near arrival. Earlier scarcity decisions remain weak.

The matched original-time-scale comparison performs better on average, so the gain cannot be attributed specifically to logarithmic time encoding. A separate planner fitted to the original data shows that useful pooled information already exists under the simulator's structural assumptions.

## The report and poster record the completed study.

The [research report](docs/policy_improvement/report.md) connects the question, experiments, explanations, and limitations. The [poster](docs/policy_improvement/poster.pdf) presents the main findings. The [decision log](docs/policy_improvement/decision_log.md) records choices made between experimental stages.

The supporting [metrics](docs/policy_improvement/evidence/final_metrics.csv), [claim ledger](docs/policy_improvement/evidence/claim_evidence.csv), [case comparisons](docs/policy_improvement/evidence/confirmation_cases.csv), and [regional comparison](docs/policy_improvement/evidence/regional_comparison.csv) preserve detailed results. The [summary](docs/policy_improvement/evidence/final_summary.json) and [evidence inventory](docs/policy_improvement/evidence/evidence_inventory.json) identify the shared numbers and models.

The findings depend on an assumed demand law and simplified inventory dynamics. They do not establish real-hotel revenue gains or a causal benefit from using explanations. Partner data will determine which assumptions can be revisited.

## Saved evidence supports reproduction.

The notebooks include executed outputs and can be read without rerunning experiments.

To rebuild the report and its evidence from saved results, use Python 3.12 with `requirements-cql.txt` installed in the project's `.venv`. Run this command from the repository root:

```sh
.venv/bin/python -m scripts.policy_improvement.report
```

This command verifies saved results and refreshes the report without retraining. Add `--refresh-diagnostics` to also rebuild experiment summaries.

Edit the notebooks directly; `scripts/run_notebook.py` executes them and saves their outputs in place.

For new experiments, run `.venv/bin/python -m scripts.policy_improvement.experiments --help`; its `confirmation` stage uses fresh confirmation data and reserved seeds. Experiment settings are in `configs/policy_improvement/`; the [decision log](docs/policy_improvement/decision_log.md) explains the experimental sequence.

Rebuilding requires local datasets and trained models under `data/` and `outputs/`. These assets are excluded from Git and are not included in a source checkout.
