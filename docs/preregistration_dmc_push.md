# Pre-registration: under disturbances, gate on the regret bound (DMC disturbance study)

Written on 2026-11-12, before any result of this experiment existed (the pilot measured fixed chunking only). The
commit that adds this file is the timestamp. Every result is reported, pass or fail.

## Hypothesis

The theory note (docs/theory_imagined_regret.md) bounds the regret of a committed action:
regret_j <= IR_j + Lip * dist(real state s_j, imagined state z_j) + slack. IR, computed before acting, covers the
first term only. Without disturbances the second term is small (ManiSkill calibration plans: PE adds almost nothing
to IR for ranking regret, AUROC 0.819 vs 0.827). Unforeseen pushes make it large, and only a signal measured while
acting (PE = |enc(s_j) − z_j|) can see it. So under pushes the gate should score both terms.

## The bound gate (BND)

BND_j = a * IR_j + b * PE_j, checked while acting: before executing plan action j (j >= 1), the robot encodes the real
state s_j, computes PE_j, and continues while the running maximum of BND stays below the threshold. a, b >= 0 are fitted
per task by non-negative least squares of the reg oracle on [IR, PE, 1] over the calibration steps (src/commit_dmc.py,
fit_bound). Thresholds and the budget controller are the same as for every gate.

## Design (code: `src/commit_dmc.py +study=push`, cluster/dmc_push_study.sh)

- Model and planner: released TD-MPC2 mt30-48M, the same as the DMC study.
- Pushes: strength 0.5, as in the pilot (docs/protocol_dmc_push_pilot.md); calibration and evaluation both pushed.
- Tasks: the 5 that qualified in the pilot: finger-turn-hard, hopper-stand, pendulum-swingup, walker-run, walker-walk.
- Arms (budget 6 actions per plan): k1, k6, random@6, IR@6, PE@6, BND@6, reg@6 (decision oracle, a ceiling).
- Calibration: 5 pushed episodes per task, seeds 6900-6904, with the decision oracles (gives IR, PE, reg).
- Evaluation: 30 episodes per arm, seeds 8100-8129, never used.

## Claims (decided now)

- **P1. Commitment is costly under pushes (replicates the pilot):** k1 − k6 > 0, 95% CI above 0.
- **P2. The bound gate beats IR alone:** BND@6 − IR@6 > 0, 95% CI above 0.
- **P3. The bound gate beats PE alone:** BND@6 − PE@6 > 0, 95% CI above 0.
- **P4. The bound gate beats fixed chunking:** BND@6 − k6 > 0, 95% CI above 0.

**Statistics.** Score = return / 1000. Differences paired by seed within each task, averaged within tasks and then
over tasks; 95% CIs from a bootstrap that resamples episodes within each task.

**Reported, not claimed:** IR@6, PE@6, reg@6 and random@6 against k6 and random@6; the fitted weights a, b per task;
achieved actions per plan; AUROC of IR, PE and BND for the reg labels on the calibration plans.

## Procedure

A smoke job (1 task, 1 calibration and 1 evaluation episode) checks the pipeline first. Its numbers are not used.
Amendments are allowed only before the full jobs start, only for problems the smoke job finds, and are committed with
a reason.
