# Pre-registration: a run-time warning light for the world model

Written on 2026-10-02, before any result of this experiment existed. The commit that adds this file is the timestamp.
Every result is reported, pass or fail.

## Problem

A robot that plans with a learned world model cannot tell when its imagination is wrong. Planning further ahead should
help, but standard TD-MPC2 does *worse* with a 12-step plan than with a 3-step plan: success drops by 0.10 (our
earlier results). Our explanation is that the planner picks the plans whose imagined future is over-optimistic.

**Question.** Can the learned Bellman audit (LBA) act as a run-time warning light, on the plan the robot is about to
execute, and does acting on the warning make the robot more successful?

## Design (code: `src/monitor.py`, analysis: `src/tools/monitor_report.py`)

- **Base planner:** MPPI with a 12-step horizon and standard scoring (H12).
- **Warning light:** at every step the monitor scores the plan H12 is about to act on, using its max over the plan's
  steps. If the score is above a threshold, the robot takes the action of a 3-step planner (H3) instead.
- **Monitors:**
  - the LBA (critic only, no extra models);
  - the raw signals A (one-step residual), B (critic disagreement), E (ELVIS-style UCB), D (dynamics ensemble) and
    M (MOBILE-style), each used as is;
  - a random warning at the same rate (control);
  - an oracle that knows the plan's true error from the simulator (ceiling).
- **LBA:** the same features, labels and trees as the detection results (`fit_lba` and `make_gbt`), fitted on the
  robot's 50 saved real episodes (seeds 2000–2049).
- **Thresholds:** set before the evaluation, from 10 closed-loop H12 episodes on separate seeds (6900–6909). For each
  monitor, the quantile of its per-step score that gives warning rates of 10%, 25% and 50%. Scores exactly at the threshold
  warn at random with the probability that keeps the rate exact, which matters if probabilities saturate at 1.0.
- **Arms**, 100 evaluation episodes each, seeds 6000–6099, the same for every arm:
  - H3; H12; constant-λ 0.8 at H12 (our best earlier planner);
  - H12 + LBA at 10%, 25% and 50%;
  - H12 + random at 10%, 25% and 50%;
  - H12 + A, B, E, D, M and oracle, each at 25%.
- **Accuracy data:** in the H12 arm, every chosen plan is also executed in the simulator from a saved state. Labels
  are the same as in the detection results: return error above the 99th percentile of real-action step-0 error
  (positive), or at or below the 90th percentile (negative).
- **Robots:** the 11 robots trained with our critic. These have the extra world models that D and M need.
  - Push s10 and s40;
  - StackCube s10, s20, s30 and s40;
  - YCB s10, s20, s30 and s40;
  - Peg s10.

  The 4 seed-50 robots repeat the experiment as a confirmation once they are trained.

## Claims (decided now)

- **W1. Accurate on the plans the robot acts on.** The LBA's mean step-stratified AUROC over robots is higher than
  that of each raw monitor (A, B, E, D, M).
- **W2. Acting on the warning helps.** success(H12 + LBA at 25%) − success(H12) > 0, with the 95% CI above 0.
- **W3. The detector matters, not just using H3 more often.** success(H12 + LBA at 25%) − success(H12 + random at
  25%) > 0, with the 95% CI above 0.
- **W4. Better than the other warning lights.** success(H12 + LBA at 25%) − success(H12 + X at 25%) has a positive
  mean for every X in A, B, E, D, M.

**Statistics.** Differences are paired by seed within each robot, averaged within robots and then over robots.
The 95% CIs come from a bootstrap that resamples episodes within each robot.

**Reported, not claimed:**
- LBA vs H3; LBA vs constant-λ;
- the warning rates of 10% and 50%;
- the share of the oracle's gain the LBA captures;
- the warning rates actually reached;
- simulator-snapshot reproducibility.

## What would count against the idea

- **W3 fails:** the warning light is no better than switching to H3 at random. That would mean the planner-time
  benefit comes from H3, not from detection.
- **W1 fails:** the LBA is not accurate on planner-chosen plans. That would mean detection on real actions does not
  transfer to the plans the planner picks.

Both outcomes will be reported.

## Procedure

1. **Smoke job** (`monitor_smoke.txt`: one robot per task, 2 episodes per arm). It checks that every arm runs,
   that snapshots reproduce the real step (max error < 1e-4), that there are no NaNs, and how long it takes. It
   produces no results.
2. **Full run** (`monitor_runs.txt`), with no code changes after the smoke job unless the smoke job fails. Any fix
   is logged here.
3. **Analysis:** `python -m src.tools.monitor_report results/mon_*.tar.gz`.
