# Results: wall-clock study 2 (faster IR scorer)

Pre-registration: docs/preregistration_wallclock2.md (commit fc671aa). 12 robots, 100 episodes per arm, seeds
6200-6299, all arms of a robot in one job on one GPU (cluster 8539297). Analysis: `python -m src.tools.wallclock_report
results/t2_*.tar.gz` (its W1-W4 lines are V1, V2, V4, V3). Every claim is reported.

## Claims

| claim | result | rule | verdict |
|---|---|---|---|
| V1 gate time / planning time | 0.082 | < 0.15 | PASS |
| V2 speedup k1 / IR@2 | 1.727 [1.692, 1.767] | CI above 1 | PASS |
| V3 time IR@2 / k2 | 1.065 [1.043, 1.088] | mean < 1.15 | PASS |
| V3 time IR@6 / k6 | 1.034 [1.026, 1.043] | mean < 1.15 | PASS |
| V4 IR@2 − k2 success | +0.023 [−0.003, +0.050] | CI above 0 | FAIL |
| V4 IR@6 − k6 success | +0.038 [+0.009, +0.068] | CI above 0 | PASS |

The batched scorer cut the gate from 11.5 ms (study 1) to 5.4 ms per plan, i.e. from 24% to 8% of a planning call.
(Planning itself measured 66.3 ms per plan here vs 47.0 ms in study 1; GPU nodes differ between runs, so absolute
ms are node-dependent; the ratios are measured within each job.)

## Arms (mean over 12 robots)

| arm | success | actions/plan | ms/step | speed vs k1 |
|---|---|---|---|---|
| k1 | 0.730 | 1.00 | 72.4 | 1.00x |
| k2 | 0.686 | 2.00 | 39.4 | 1.84x |
| k4 | 0.617 | 3.85 | 23.4 | 3.10x |
| k6 | 0.538 | 5.56 | 18.0 | 4.04x |
| IR@2 | 0.709 | 2.02 | 41.9 | 1.73x |
| IR@4 | 0.657 | 3.97 | 24.4 | 2.98x |
| IR@6 | 0.577 | 5.76 | 18.6 | 3.90x |

## Reported (not claimed)

- IR@4 − k4: +0.040 [+0.012, +0.069].
- k1 − IR@2 +0.021 [−0.005, +0.048]; k1 − IR@4 +0.073; k1 − IR@6 +0.153.
- Pooled over studies 1 and 2 (24 robot-runs, disjoint seeds): IR@2 − k2 +0.022 [+0.004, +0.041];
  IR@4 − k4 +0.036 [+0.015, +0.057]; IR@6 − k6 +0.038 [+0.018, +0.060].

## Reading

The overhead problem from study 1 is solved: the gate costs 8% of planning, IR@b runs within 3-7% of the time of k_b,
and IR@2 is 1.73x faster than replanning every step. IR beats fixed chunking at budgets 4 and 6 in both studies. At
budget 2 the effect (+0.02) is again positive but its CI touches 0 in each study alone (V4 at b=2 fails); pooling the
two studies, reported not claimed, gives a CI above 0. At budget 2 IR@2 loses 0.021 success to k1 (CI includes 0)
while running 1.73x faster.
