# Baseline diagnosis and controlled comparison

This document records how the baseline findings were obtained and what can be concluded from them. The [README](../README.md) explains why these questions follow from the initial result; the [environment specification](environment_spec.md) defines the simulated world.

## Questions and fixed inputs

The analysis asks where the saved CQL policy loses revenue, how those losses relate to training support, and whether removing the conservative penalty improves decisions.

The original inputs are `data/simulated/baseline/` (collection seed 42) and `outputs/baseline/cql/`. The analyses reuse the original CQL checkpoints. Demand, state definition, five-price menu, reward, and terminal rules remain fixed. Dataset checksums and checkpoint-to-policy agreement establish that the comparisons use the intended baseline.

## Locate the decision losses

[The policy-gap notebook](../notebooks/baseline_analysis/01_policy_gap.ipynb) uses exact calculations under the simulator. It does not retrain policies or collect evaluation rollouts.

[policy_gap.py](../hotel_pricing/policy_gap.py) computes optimal values backward, then propagates state visitation probabilities forward under each saved CQL policy. At a state `s`, local regret is:

```text
V_star(s) - Q_star(s, policy(s))
```

Here, `Q_star` means taking the policy's chosen action once and then continuing optimally. Weighting this regret by the policy's own state visitation probabilities and summing reproduces its initial expected revenue gap. Using optimal-policy visitation weights would not give the same decomposition.

For seed 0:

| Diagnostic | Result |
| --- | ---: |
| Initial expected revenue gap | 37.4073 |
| Share in the final 30 days | 46.6% |
| Share with one or two rooms remaining | 48.4% |
| Share where the optimal action has no exact state–action training examples | 24.6% |
| Share where the optimal action has at least 20 examples | 23.2% |

The time and inventory groups overlap and must not be added. Seed 1 has the same decomposition; seed 2 is similar. Counts measure empirical support, not causal effects of data scarcity. State tables are saved under `outputs/baseline/gap_diagnosis/`.

## Isolate the conservative penalty

[The CQL–DQN notebook](../notebooks/baseline_analysis/02_cql_vs_offline_dqn.ipynb) reads the saved controlled comparison. The runner is [baseline_comparison.py](../hotel_pricing/baseline_comparison.py); its protocol is [baseline_comparison.json](../configs/baseline_comparison.json).

Both methods use the same original training data and these settings:

| Setting | Value |
| --- | --- |
| Network | Two 64-unit ReLU hidden layers, five action values |
| Training seeds | 0, 1, 2 |
| Training updates / batch size | 20,000 / 256 |
| Optimizer / learning rate | Adam / 0.0003 |
| Target refresh | Every 100 updates |
| State scaling | Time divided by horizon; rooms divided by capacity |
| Reward scaling / discount | Revenue divided by 90 / gamma 1 |
| CQL coefficient | 0.1 for CQL; 0 for offline DQN |
| Checkpoint selection | Final update for every seed |

DQN never collects online training experience. The runner checks the learning and exact-evaluation implementations against recorded baseline source hashes. It saves the protocol and input checksums before training. There is no tuning or best-seed selection in this comparison.

Every policy is evaluated on 1,000 fresh episodes with matched sale randomness, starting at environment seed 12000000. Action randomness starts at 14000000. Exact expected revenue is also calculated under the unchanged simulator. Full results are saved in `outputs/baseline/comparison/`.

## Results and interpretation

| Training seed | CQL expected revenue | Offline DQN expected revenue |
| --- | ---: | ---: |
| 0 | 851.00 | 849.45 |
| 1 | 851.00 | 829.11 |
| 2 | 851.08 | 815.81 |

The best fixed price achieves 851.00; the dynamic optimum achieves 888.41. CQL performs better for all three matched seeds, although DQN seed 0 is close. This experiment does not support removing conservatism as a remedy for the baseline's limited adaptation.

Exact expectations have no rollout sampling uncertainty conditional on the simulator and policy. Paired rollout intervals describe episode sampling uncertainty for fixed models; they do not describe variation across training datasets. Three training seeds on one dataset do not establish a general algorithm ranking. Revenue excludes operating costs.

## Reproduction and provenance

Run commands from the repository root using Python 3.12. Reuse the existing `.venv`, or create an environment and install the dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cql.txt
```

The analysis requires the original dataset in `data/simulated/baseline/`, CQL checkpoints and results in `outputs/baseline/cql/`, and comparison results in `outputs/baseline/comparison/`. These artifacts are excluded from Git and must be available locally; a source checkout alone does not include them.

With the saved artifacts present, execute the notebooks and verification tests:

```sh
.venv/bin/python scripts/run_notebook.py notebooks/baseline_analysis/01_policy_gap.ipynb
.venv/bin/python scripts/run_notebook.py notebooks/baseline_analysis/02_cql_vs_offline_dqn.ipynb
.venv/bin/python -m unittest discover -s tests -v
```

The notebook runner saves executed notebooks and HTML reading copies. The notebooks reuse saved models without retraining.

If only the comparison results are missing, generate them from the original dataset and CQL checkpoints:

```sh
.venv/bin/python -m hotel_pricing.baseline_comparison
```

This command re-evaluates CQL and trains three offline DQN models for 20,000 updates each. Skip it when comparison results already exist. The writer refuses to overwrite saved runs; a separate run requires a new `--output` directory. Regenerated datasets or checkpoints must not be substituted silently for the original baseline.

[The provenance record](provenance/baseline.json) stores learning/evaluation source hashes, artifact checksums, and the mapping from historical folders to current locations. Saved manifests and results retain their original metadata. The small collection fixture in `tests/fixtures/baseline_collection.json` was generated by the original collector before the duplicate code tree was removed, and verifies unchanged default collection behavior.

Subsequent changes to collection or learning should receive a separate protocol and new artifact locations. The current dataset and evaluation have already been inspected; fresh final evaluation seeds are needed for later experiments.
