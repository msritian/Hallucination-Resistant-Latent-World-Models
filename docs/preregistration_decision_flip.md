# Pre-registration: training-free decision-flip signals

Written on 2026-11-10, before any result of this experiment or of the confirmation study existed. The commit that adds
this file is the timestamp. Every result is reported, pass or fail.

## Idea

The commitment study suggests that the errors that matter for control are those that change the decision. Two
signals estimate that directly from parts TD-MPC2 already has, with no labels and no training (code:
`flip_scores` in `src/commit.py`). For each step j of the chosen plan, judging plan action j+1:

- Every dynamics head (the main one plus the 5 auxiliary heads) imagines the plan on its own, giving several
  plausible imagined states z^m_{j+1}.
- **PD (policy disagreement):** mean over heads of |pi(z^m_{j+1}) − pi(z^0_{j+1})|. Would the robot's policy act
  differently in the plausible states? This is disagreement in action space; the ensemble signal D measures it in
  state space.
- **IR (imagined regret):** mean over dynamics heads m and critic heads h of
  max(0, Q_h(z^m_{j+1}, pi(z^m_{j+1})) − Q_h(z^m_{j+1}, a_{j+1})). How much worse is the stale plan action than a fresh
  choice, across plausible states and critics? This is a label-free estimate of the reg oracle.

## Design

Run inside the confirmation study's jobs (docs/preregistration_commitment_confirm.md, amendment 1): same 12 fresh
robots, seeds, calibration, budget controller and statistics. Arms PD@4 and IR@4 are added there; all other arms
(random@4, act@4, reg@4, D@4, LBA@4, ...) are shared.

## Claims (decided now)

- **F1. A training-free decision signal pays off under commitment:** success(IR@4) − success(random@4) > 0, 95% CI
  above 0.
- **F2. Decision space beats state space:** success(PD@4) − success(D@4) > 0, 95% CI above 0.
- **F3. Better than the return-error audit:** success(IR@4) − success(LBA@4) has a positive mean.

**Reported, not claimed:** PD@4 − random@4; IR@4 and PD@4 against DBA@4, Aa@4, PE@4 and act@4; step-stratified AUROC
of PD and IR for both targets (return-error labels and decision labels) on the calibration plans.
