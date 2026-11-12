# DMC headroom pilot: results

Protocol and decision rule: docs/protocol_dmc_headroom_pilot.md (commit 964dd42, written before any pilot data).
Released mt30-48M model, fixed chunking only, 30 episodes per arm, seeds 7000-7029, score = return / 1000.
Differences are paired by seed; 95% CI from a bootstrap over episodes.

| Task | k1 | k4 | k6 | k8 | k1 - k4 | k1 - k6 [95% CI] | k1 - k8 | Qualifies (k1 - k6 >= 0.05) |
| :-- | --: | --: | --: | --: | --: | :-- | --: | :-- |
| cheetah-run | 0.134 | 0.117 | 0.106 | 0.083 | +0.018 | +0.028 [+0.000, +0.058] | +0.051 | no |
| finger-turn-hard | 0.951 | 0.903 | 0.833 | 0.833 | +0.048 | +0.118 [+0.031, +0.228] | +0.119 | yes |
| fish-swim | 0.473 | 0.470 | 0.468 | 0.453 | +0.003 | +0.006 [-0.047, +0.053] | +0.021 | no |
| hopper-stand | 0.619 | 0.573 | 0.635 | 0.622 | +0.045 | -0.017 [-0.085, +0.041] | -0.003 | no |
| walker-run | 0.519 | 0.499 | 0.445 | 0.368 | +0.019 | +0.074 [+0.036, +0.106] | +0.150 | yes |

**Decision (as pre-committed):** 2 of 5 tasks qualify, fewer than 3, so no DMC gate study is run. DMC is reported as a
domain where committing to the plan is nearly free for this model: at 4 actions per plan the cost is at most 0.05 on
every task, and at 6 actions it is large only on finger-turn-hard and walker-run.

Exploratory note (not part of the rule): the cost grows with k on cheetah-run and walker-run (k1 - k8 = +0.051,
+0.150), so headroom exists in DMC only at long blind horizons. The ManiSkill results (commitment costs 0.25 at
k = 4) are where gating matters.
