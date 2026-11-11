# Pre-registration: which hallucinations matter for control? (commitment study, confirmation)

Written on 2026-11-10, before any result of this experiment existed. The commit that adds this file is the timestamp.
Every result is reported, pass or fail.

## Background

In the commitment study (docs/results_commitment.md, 10 robots) the robot executes several planned actions blind.
Committing hurts (success 0.620 at k = 1, 0.492 at 4 actions per plan), but a perfect detector of our hallucination
label (imagined vs real return error, "oracle") beat random gating by only +0.029 [−0.004, +0.063]. Two other gates
did twice as well (exploratory): the label-free anchored residual Aa (+0.054 [+0.021, +0.086]) and the prediction-error
check PE (+0.056 [+0.022, +0.089]).

**Hypothesis.** The return-error label is the wrong target for control: an imagined value can be far off while the
chosen action is still right, and a small error can change the action. The errors that matter are those that change
the decision.

## Design (code: `src/commit.py --study confirm`, analysis: `src/tools/commit_report.py`)

Same planner, commitment rule, budget controller (amendment 1 of the main study) and statistics as the main study, at
budget 4 actions per plan only. New gates:

- **Decision oracles** (simulator, a ceiling): for each plan and each j = 1..7, execute the first j plan actions in a
  simulator copy, replan from the real state reached (a second planner, warm-started as the robot would be), and
  compare with the stale plan action j:
  - **act**: |fresh action − plan action| (would a fresh plan act differently?);
  - **reg**: Q(real state, fresh action) − Q(real state, plan action) (the critic's regret of the stale action).
  The gate commits while these stay below the threshold, exactly like the other gates.
- **DBA, the decision-aware Bellman audit:** the LBA's features and trees, trained to predict the act oracle instead
  of the return error. Labels: top quarter of act (positive) vs bottom half (negative), from 20 training episodes
  per robot (seeds 6800–6819). Uses only the plan at run time, like the LBA.

**Arms** (100 evaluation episodes each, seeds 6000–6099): k1, k4, random@4, oracle@4 (return error), act@4, reg@4,
DBA@4, LBA@4, Aa@4, D@4, M@4, PE@4. Calibration: 10 episodes (seeds 6900–6909). LBA: fitted as before.

**Robots: 12, none used in any commitment experiment.**
- Push, StackCube and YCB seed 50, using their saved real episodes;
- the 9 training-maturity robots at 100% of training: Push, StackCube and YCB, seeds 60–62. These have no saved
  episodes, so 50 real episodes are collected on seeds 2000–2049, the protocol used for every other robot.

Peg seed 50 is excluded for the same reason as before (near-zero success).

## Claims (decided now)

- **K1. Decision-relevant errors are costly:** success(act@4) − success(random@4) > 0, 95% CI above 0.
- **K2. They are not the return-error hallucinations:** success(act@4) − success(oracle@4) > 0, 95% CI above 0.
- **K3. The exploratory gates replicate:** success(Aa@4) − success(random@4) > 0 and success(PE@4) − success(random@4)
  > 0, each with the 95% CI above 0.
- **K4. A decision-aware audit works before acting:** success(DBA@4) − success(random@4) > 0 with the 95% CI above 0,
  and success(DBA@4) − success(LBA@4) has a positive mean.

**Statistics.** As before: differences paired by seed within each robot, averaged within robots and then over robots;
95% CIs from a bootstrap that resamples episodes within each robot.

**Reported, not claimed:** reg@4 against random and act; LBA, D, M against random; DBA against Aa and PE; k1 − k4; each
score's accuracy (step-stratified AUROC) on the calibration plans for both targets: return-error labels and
decision labels (act top quarter vs bottom half); achieved actions per plan.

## Procedure

A smoke job (1 robot, 2 episodes per arm, small training and calibration sets) checks the pipeline first. Its numbers
are not used. Amendments are allowed only before the full jobs start, only for problems the smoke job finds, and are
committed with a reason.
