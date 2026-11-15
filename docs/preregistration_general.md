# Pre-registration: does IR generalise? (stock TD-MPC2 and new tasks)

Written on 2026-11-13, before any result of this experiment existed (the new-task models are not trained yet). The
commit that adds this file is the timestamp. Every result is reported, pass or fail.

## Why

Every positive IR result so far comes from our grounded TD-MPC2 robots (with an ensemble of dynamics heads) on three
ManiSkill tasks (PushCube, StackCube, PickSingleYCB). Two open questions:
1. Does IR work for unmodified TD-MPC2, which has a single dynamics head and no grounding? (The single-head ablation
   IR1 already worked on grounded robots, frontier study G4.)
2. Does it work on tasks it was never developed on?

## Design (code: `src/commit.py --study general`, cluster/commit_general.sh)

Per robot: 50 real episodes (seeds 2000-2049) fit the LBA, which fixes L as in every earlier study; 10 calibration
episodes (seeds 6900-6909) give every gate's thresholds; then the arms, 100 evaluation episodes each, seeds 6300-6399
(never used). IR uses the batched scorer of wall-clock study 2 (checked against the original scorer in every job).
Same planner, commitment rule and budget controller as before.

- **Arms:** k1, k2, k4, k6, random@4, IR@2, IR@4, IR@6, PE@4.
- **Part S, stock TD-MPC2 (10 robots, none used in any commitment study):** the stock models of PushCube s40, s50;
  StackCube s20, s30, s40, s50; PickSingleYCB s20, s30, s40, s50. One dynamics head, so IR averages over critic heads
  only.
- **Part N, new tasks (12 robots):** grounded TD-MPC2 trained now with the recipe of the maturity study on 4 tasks
  never used in this project's commitment work: PickCube-v1 (500k steps), PokeCube-v1, PullCubeTool-v1,
  LiftPegUpright-v1 (1M steps each); seeds 70, 71, 72 (cluster/newtask_train_runs.txt).
- **Inclusion rule (both parts):** a robot is analysed if the success rate of its 50 collection episodes is at least
  0.20 (Peg seed 50 was excluded from earlier studies for the same reason). Decided before any of these numbers
  exist.

## Claims (decided now; each part tested separately, over its included robots)

- **S1 / N1. IR beats random gating:** success(IR@4) − success(random@4) > 0, 95% CI above 0.
- **S2 / N2. IR beats fixed chunking at equal compute:** success(IR@b) − success(k_b) > 0, 95% CI above 0, for
  b = 4 and for b = 6.

**Statistics.** As in all earlier studies: differences paired by seed within each robot, averaged within robots and then
over robots; 95% CIs from a bootstrap that resamples episodes within each robot.

**Reported, not claimed:** IR@2 − k2; PE@4 against random@4, k4 and IR@4; k1 − k_b; actions per plan; per-task
results; the pooled analysis over both parts; robots excluded by the inclusion rule, with their success.

## Procedure

A smoke job (1 stock robot, 2 episodes per arm) checks the pipeline first; Part N gets its own smoke job once its models
exist. Smoke numbers are not used. Amendments only before the full jobs of a part start, only for problems a smoke job
finds, committed with a reason.
