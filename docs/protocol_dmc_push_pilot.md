# DMC disturbance pilot: protocol and decision rule

Written on 2026-11-12, before any pilot data. The commit that adds this file is the timestamp.

## Why

Without disturbances, committing to the plan is nearly free on DMC for the released mt30-48M model
(docs/results_dmc_headroom_pilot.md), so no gate can help. The regret bound (docs/theory_imagined_regret.md) says a
committed action's regret is at most IR + Lip * dist(real state, imagined state) + slack. Without disturbances the
imagined states stay close to reality (on ManiSkill's calibration plans, prediction error PE adds almost nothing to
IR for ranking regret: AUROC 0.819 vs 0.827). Unforeseen external pushes make the distance term large: IR cannot see
them in advance, PE sees them right after they happen. A disturbed domain is where the bound predicts that the
before-acting signal IR must be combined with the while-acting signal PE.

This pilot only measures whether pushes make commitment costly. It runs no gates.

## Disturbance

Seeded per episode (src/commit_dmc.py, DMCSim.set_pushes): at each env step a push starts with probability 0.02
(about 10 pushes per 500-step episode) and lasts 5 env steps; the force is alpha times the robot's weight in a
random 3D direction, applied to body 1 (the root body). The schedule depends only on the episode seed, so every arm
and every simulator copy meets the same pushes.

## Design

- Model: released TD-MPC2 mt30-48M, the same planner as the DMC study.
- Tasks: the 10 of the DMC study.
- Strengths: alpha = 0.5 and alpha = 1.0.
- Arms: fixed chunking k = 1, 4, 6 only.
- Seeds: 20 episodes per arm, seeds 7100-7119 (never used by any study).

## Decision rule (fixed now)

- A task qualifies at a strength if mean(k1 − k6) >= 0.05 and its k1 score is at least 0.10 (still solvable).
- The disturbance study uses the smaller strength at which at least 4 tasks qualify, on those tasks, with fresh
  evaluation seeds (8100 onward). Its arms and claims are pre-registered separately, before it runs.
- If neither strength gives 4 qualifying tasks, DMC is reported as robust to commitment even under pushes and no
  disturbance study is run.
