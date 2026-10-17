# Hotel Pricing

## Repository structure

```text
├── hotel_pricing/          # Simulator, learning, and evaluation
├── configs/                # Environment and experiment settings
├── notebooks/
│   ├── baseline/           # Booking analysis, simulation, and initial learning
│   └── baseline_analysis/  # Policy diagnosis and algorithm comparison
├── data/                   # Source metadata, bookings, and simulated episodes
├── outputs/                # Saved models, results, tables, and figures
├── docs/                   # Assumptions, experiment details, and provenance
├── scripts/                # Data download and notebook execution
├── tests/                  # Implementation and regression checks
└── requirements-*.txt      # Python dependencies
```

## Pricing is learned from experience.

This capstone with SAS investigates how pricing decisions can be learned directly from experience, rather than relying on a separate demand forecast before selecting prices. The central question is whether a policy can use the outcomes of past decisions to choose prices as time and available inventory change.

While partner data undergoes confidentiality checks and anonymization, the project uses simulation to develop the learning workflow and investigate its limitations.

## Booking data informs the setting.

Public hotel reservations provide a starting point for understanding lead times, room types, rates, cancellations, and stays. These patterns help define a manageable pricing setting.

However, reservations do not capture every offered price, available inventory at each decision, or shoppers who did not book. The data therefore informs the simulator rather than serving as a complete history for learning pricing decisions.

The [booking analysis](notebooks/baseline/01_hotel_booking_patterns.ipynb) establishes this groundwork. The [environment specification](docs/environment_spec.md) separates observed patterns from modeling assumptions.

## Simulation makes decisions testable.

The baseline represents a single arrival date, room type, and stay night. A policy observes time and rooms remaining, chooses from a fixed price menu, and receives revenue when a room sells. The simulator records both sales and no-sale days.

A deliberately small environment allows the best achievable policy to be calculated exactly. The learner uses recorded experience, while the known demand mechanism supports evaluation. This provides a controlled way to assess pricing decisions before introducing richer demand patterns.

The [simulator notebook](notebooks/baseline/02_simulator_and_offline_data.ipynb) describes the environment and collection of offline training data.

## The baseline reveals limited adaptation.

Conservative Q-Learning (CQL) learns from a fixed dataset of simulated episodes. Its decisions are compared with the data-collection policy, fixed prices, and the simulator-informed optimum.

The learned policy improves on the collection policy but behaves largely like a fixed-price rule. This establishes a working pipeline while motivating a closer examination of why the policy does not adapt effectively to time and inventory.

The [CQL notebook](notebooks/baseline/03_offline_cql.ipynb) presents the experiment, and the [end-to-end summary](notebooks/baseline/04_end_to_end_demo.ipynb) connects the initial workflow.

## Diagnosis guides the investigation.

The analysis preserves the original data and models and locates the decisions responsible for the remaining revenue gap. Comparing these losses with training support identifies potentially important gaps in experience without assuming that limited coverage caused the errors.

A controlled comparison then tests whether removing CQL's conservative penalty improves performance. Offline DQN uses the same data, architecture, training budget, and seeds. The comparison does not support removing the penalty as a remedy in this setting.

The [policy-gap diagnosis](notebooks/baseline_analysis/01_policy_gap.ipynb) and [CQL–DQN comparison](notebooks/baseline_analysis/02_cql_vs_offline_dqn.ipynb) document these investigations. The [experiment record](docs/baseline_comparison.md) retains the detailed controls and findings.

## Conclusions depend on the simulated world.

The experiments provide evidence about learning under explicit demand assumptions. They do not establish performance on partner data or demonstrate real-world revenue gains. As that data becomes available, its contents will determine which assumptions can be revisited and which parts of the workflow can carry over.
