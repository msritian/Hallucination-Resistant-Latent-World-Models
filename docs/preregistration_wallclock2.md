# Pre-registration: wall-clock study 2 (a faster IR scorer)

Written on 2026-11-13, before any result of this experiment existed. The commit that adds this file is the timestamp.
Every result is reported, pass or fail. This is a new study; the failures of wall-clock study 1
(docs/results_wallclock.md: W1, W3 at budget 2, W4 at budget 2) stay on the record.

## Why

In study 1 the lean IR gate cost 24% of a planning call (11.5 ms vs 47.0 ms). The cost is kernel-launch latency:
the scorer rolls each of the M dynamics heads separately through all 12 plan steps (about 60 small MLP calls),
although IR uses only the first L = 7 imagined states, and it calls the critic twice.

## The faster scorer (`ir_fast` in src/commit.py)

Same IR, computed with: all dynamics heads in one batched matrix multiply per step (their weights stacked; they share
one architecture), only the L steps used, and one critic call for the policy and plan actions together. On random
heads built with TD-MPC2's layers the batched rollout matches the per-head rollout to 2e-7. Each job first checks on
5 plans that IR matches the original scorer (max relative difference < 1e-3; float rounding only) and stops
otherwise.

## Design (code: `src/commit.py --study timing2`, cluster/commit_timing2.sh)

Identical to study 1 except the scorer and the seeds: the same 12 robots, arms k1, k2, k4, k6, IR@2, IR@4, IR@6, all
arms of a robot in one job on one GPU, thresholds and L reused unchanged, 100 episodes per arm, seeds 6200-6299 (never
used), per-plan planning and gate times with CUDA synchronisation.

## Claims (decided now)

- **V1. The gate is cheap:** mean over robots of (total gate time / total planning time) over the IR arms < 0.15.
- **V2. IR saves real time:** time per step of k1 / time per step of IR@2 > 1, 95% CI above 1.
- **V3. The gain is not bought with hidden compute:** for b = 2 and 6, mean over robots of
  time per step of IR@b / time per step of k_b < 1.15.
- **V4. IR beats fixed chunking:** success(IR@b) − success(k_b) > 0 with the 95% CI above 0, for b = 2 and b = 6.

Statistics as in study 1 (src/tools/wallclock_report.py; its W1-W4 lines are V1, V2, V4, V3 here).

**Reported, not claimed:** IR@4 vs k4; k1 − IR@b; speedups of every arm over k1; actions per plan; planning and gate
ms per plan; the pooled IR@2 − k2 over studies 1 and 2 (24 robot-runs).

## Procedure

A smoke job (1 robot, 2 episodes per arm) first. Amendments only before the full jobs, only for problems the smoke
job finds, committed with a reason.
