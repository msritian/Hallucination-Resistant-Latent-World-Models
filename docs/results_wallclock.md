# Wall-clock study: results

Pre-registration: docs/preregistration_wallclock.md (commit fc1fb97). 12 robots, 100 episodes per arm, fresh seeds
6100-6199, all arms of a robot in one job on one GPU. Lean IR scorer checked against the old one: max difference 0.
Analysis: `python -m src.tools.wallclock_report results/tm_*.tar.gz`.

| Arm | Success | Actions per plan | ms per step | Speed vs k1 |
| :-- | --: | --: | --: | --: |
| k1 | 0.760 | 1.00 | 51.8 | 1.00x |
| k2 | 0.710 | 2.00 | 28.4 | 1.82x |
| k4 | 0.640 | 3.85 | 16.9 | 3.06x |
| k6 | 0.552 | 5.56 | 13.0 | 3.99x |
| IR@2 | 0.731 | 2.01 | 34.1 | 1.52x |
| IR@4 | 0.672 | 3.96 | 19.6 | 2.65x |
| IR@6 | 0.591 | 5.76 | 14.9 | 3.48x |

Per plan: planning 47.0 ms, IR gate 11.5 ms.

## Claims

| Claim | Result | Verdict |
| :-- | :-- | :-- |
| W1 gate time / planning time < 0.15 | 0.243 | FAIL |
| W2 speedup k1 / IR@2, CI above 1 | 1.523 [1.493, 1.559] | PASS |
| W3 IR@2 − k2 > 0, CI above 0 | +0.021 [−0.006, +0.047] | FAIL |
| W3 IR@6 − k6 > 0, CI above 0 | +0.038 [+0.008, +0.068] | PASS |
| W4 time IR@2 / k2 < 1.15 | 1.197 [1.170, 1.223] | FAIL |
| W4 time IR@6 / k6 < 1.15 | 1.146 [1.136, 1.157] | PASS |

Reported: IR@4 − k4 +0.033 [+0.004, +0.061]; k1 − IR@2 +0.029 [+0.003, +0.054]; k1 − IR@4 +0.087;
k1 − IR@6 +0.169.

## What it means

- IR really saves time: replanning every step is 1.52x slower than IR@2, which loses 0.029 success.
- The lean gate costs 24% of a planning call, not under 15%. That overhead is why IR@2 is 20% slower per step than
  k2 (W4 fails at budget 2), and at budget 2 the success gain over k2 (+0.021) is smaller than in the frontier study
  (+0.032) and its CI includes 0 on these seeds.
- At budgets 4 and 6, IR beats fixed chunking on fresh seeds (+0.033, +0.038), at most 15% more time per step.
- The gate cost is latency, not arithmetic: the scorer re-encodes the observation and rolls every dynamics head
  through all 12 plan steps one small kernel at a time, although only the first 8 imagined states are used and the
  planner has already encoded the observation. Removing that is engineering, not a change to IR; it is not claimed
  here.
