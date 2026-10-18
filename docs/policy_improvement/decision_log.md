# Learning follow-up decisions

The executable specification is `configs/policy_improvement/training.json`; the output protocol records source hashes, package versions, seeds, and hashes of protected historical artifacts. These experiments run on `xai-policy-audit` without changing the capstone baseline or the completed coverage study.

## Correctness and signal checks precede new interventions.

The transition audit verifies all three original splits, reward scaling, state normalization, action indexing, Boolean terminal masks, and terminal targets under extreme next-state values. No implementation defect was identified. Historical target networks were not saved, so historical residuals explicitly use the online network as a bootstrap proxy.

At `(1, 1)`, the original data contains only 37 records, including three price-55 trials with no sales. These observations do not establish that the learner should infer the true price ordering locally. The terminal positive control therefore uses 10,000 independent observed outcomes per action and 2,000 separate evaluation outcomes per action. Both empirical objectives (alpha 0 and 0.1) can be fitted by the same network architecture to within 0.001; their selected action is price 55. The ordinary minibatch learner also selects 55 after 2,000 updates, with nonzero estimation error.

The learned-demand control pools original training trials by price and plans using the estimated probabilities and known inventory dynamics. Its exact revenue is 888.373761, versus the simulator optimum 888.410341. This demonstrates useful signal under a price-only stationary structural model, not successful CQL learning or oracle-free evaluation.

## Replay is tested before adding experience.

Four development trajectories use seeds 0 and 1, with uniform replay or exactly 32 late-window and 224 earlier records per batch. Sampling retains both sales and no-sales; the TD and conservative losses use the same batch. All other baseline settings are held fixed. The first endpoint is 20,000 updates. If neither configuration qualifies, the same trajectories continue to the fixed 100,000-update endpoint; 50,000 is diagnostic only. Full online, target, optimizer, replay-count, and RNG states are saved. Numerical tests verify exact continuation and equivalence of uniform training to the original learner.

The candidate rule is positive improvement in both development seeds and at least 10% mean recovery of the ordinary-20k reference gap. Selection applies to a configuration, never a favorable seed or epoch. Confirmation seeds 10, 11, and 12 are unused and reserved. Optional development stages stop once a candidate qualifies.

## The 20k endpoint motivates matched continuation.

Uniform replay returns 851.003070 in both development seeds. Late-window replay returns 851.068935 and 851.007705: both gains are positive, but mean gap recovery is only about 0.094%. Terminal Q90 falls from 1.59652/1.23556 to 0.08888/0.01405. Calibration improves without a useful policy improvement. Before viewing longer-run outcomes, the decision is to continue all four original trajectories to 100,000 updates, retaining the fixed endpoint and all saved state.

## Sparse outcomes justify the fresh-data comparison.

At the fixed 100k endpoint, uniform replay remains 851.003070 in both seeds; late-window replay returns 851.003238 and 851.226111. Neither configuration approaches the advancement threshold. Sparse local outcomes remain credible because repeating existing records does not create independent sales observations. Before training either new arm, freeze 150,000 familiar-pair probes and 150,000 balanced probes (100 per each of 1,500 pairs), with collection namespace 61000000 and outcome seed 61000001. Both arms retain all no-sales and use the unchanged transition kernel. Both train at 100,000 updates with the same 32/224 replay. Original-data late replay at 100k is the information control; uniform replay at 20k and 100k remains the primary and compute control.

## Balanced probes improve revenue but leave a representation question.

Fresh familiar probes return 852.950416 and 851.336194. Balanced probes return 854.232201 and 854.218125, about 8.61% mean gap recovery, below the 10% advancement criterion. The original late-window data covers 602 pairs and 141 sales; familiar probes add 7,355 sales at those same 602 pairs, while balanced probes add 8,689 sales across all 1,500 pairs. Counts alone do not imply accurate per-pair estimates.

Both balanced models still rank 90 above 55 at `(1, 1)`, unlike the adequately supported terminal-only fits. Mechanics and the simple fit are sound, while shared full-task predictions remain poor near the deadline. Before observing the next outcomes, choose exactly one method change: `log1p(days)/log1p(330)` in place of `days/330`, retaining `rooms/10`, the two-input architecture, balanced development dataset, 32/224 replay, alpha 0.1, and fixed 100k endpoint. Seeds remain 0 and 1. The balanced linear-input runs are the matched method control. No alpha sweep or further method settings will be opened in this study.

## The selected configuration is frozen before confirmation.

Log-time encoding returns 861.359703 and 859.940380, with 25.79% mean gap recovery and positive improvement in both development seeds. It qualifies as a configuration; neither seed or intermediate checkpoint is selected individually. Stop opening development stages.

Freeze balanced probes, log-time input, alpha 0.1, 32/224 replay, and 100,000 updates. Regenerate 150,000 balanced probes under the reserved fresh collection namespace 62000000 (outcome stream 62000001), shared across confirmation seeds 10, 11, and 12. The ordinary uniform trajectories provide both 20k primary references and 100k compute controls. Three additional linear-time trajectories on the same fresh probe dataset isolate the input encoding contrast. No familiar-probe confirmation is run; any balanced-versus-familiar interpretation remains development evidence. The success rule remains at least 10% aggregate gap recovery and positive paired differences against uniform-20k in all three seeds. Do not retune after confirmation.

## The frozen package clears the confirmation threshold.

Candidate revenues are 866.367381, 869.805730, 857.233993 for seeds 10, 11, and 12. All exceed their ordinary-20k references (851.003070); the mean is 864.469035, recovering 36.00% of the reference gap. No candidate settings are changed after observing these outcomes. The prespecified linear-input component controls continue to completion before the representation interpretation is finalized.

## The encoding-specific benefit does not confirm.

Linear-time controls return 867.757344, 869.244643, 861.029090. Log-minus-linear paired differences are -1.389963, +0.561086, -3.795098; the log-time candidate is better in only one of three seeds. The linear control mean (866.010359) exceeds the frozen candidate mean (864.469035). Thus the package improves over ordinary training, but the development-specific encoding advantage does not replicate consistently. Do not relabel the linear control as a newly selected candidate after confirmation. Report it transparently as a strong prespecified control, and retain both configurations. The study stops without further tuning.
