# Pre-registration: does imagined regret hold across commitment budgets and against the newest baselines? (frontier study)

Written on 2026-11-11, after the confirmation results (docs/results_commitment_confirm.md) and before any result of
this experiment existed. The commit that adds this file is the timestamp. Every result is reported, pass or fail.

## Background

In the confirmation study (12 robots, budget 4 actions per plan) the training-free imagined-regret gate IR beat random
gating by +0.091 [+0.062, +0.121] and the learned Bellman audit by +0.055, and it was statistically indistinguishable
from the simulator decision oracle act (−0.012 [−0.040, +0.018]). All of this was at one budget, against our own
baselines only. Three questions are open:

1. **Generality across compute.** Does IR help at other budgets (2 and 6 actions per plan), and does it beat simply
   executing a fixed number of actions at the same planning compute?
2. **Newest baselines.** Two 2026 signal families transfer to TD-MPC2:
   - **FI, latent-flow instability:** Hansen & Wang (2606.27326) predict hallucinations in a video world model from
     flow instability, round-trip residuals and inter-seed variance. TD-MPC2 has no decoder, so round-trip residuals
     do not transfer, and inter-seed variance is our ensemble score D (already reported). The transferable analogue
     of flow instability is the second difference of the imagined latent trajectory,
     FI[j] = |z_{j+2} − 2 z_{j+1} + z_j| (main dynamics head).
   - **RS, rank instability:** "Planners Exploit Deterministic Error" (2611.11780) attributes TD-MPC2 planning failures
     to errors that change the ranking of candidate plans. RS[j]: take the chosen plan and the 15 best other candidates
     of the planner's final MPPI pool; from each dynamics head's imagined state z^m_{j+1} score every candidate's
     remaining actions with the planner's own value estimate; RS[j] = mean over heads of the share of candidates that
     beat the chosen plan.
3. **Is the ensemble needed?** IR averages over 5 dynamics heads. **IR1** uses the main head only (all critic heads), so
   it needs nothing beyond a standard TD-MPC2 agent.

## Design (code: `src/commit.py --study frontier`, analysis: `src/tools/commit_report.py`)

Same robots (the 12 of the confirmation study), evaluation seeds (6000–6099, 100 episodes), planner, commitment rule,
budget controller and statistics as the confirmation study. Each job resumes the robot's confirmation directory: the
same LBA and DBA, the same thresholds for every existing score at every commitment level, and the same budget-4 arms,
so all arms are paired by seed. New calibration (seeds 6900–6909, 10 episodes, same as before) is run only for the new
scores IR1, FI, RS.

**New arms per robot (19):**
- budgets 2 and 6: fixed k2 and k6; random, oracle (return error), act, reg, IR, LBA, PE at each budget (16);
- budget 4: IR1, FI, RS (3).

Together with the confirmation arms this gives success against planning compute (1, 2, 4, 6 actions per plan) for
fixed k, random, oracle, act, reg, IR, LBA and PE.

## Claims (decided now)

- **G1. IR generalises across budgets:** success(IR@b) − success(random@b) > 0 with the 95% CI above 0, for b = 2 and
  for b = 6 (both required).
- **G2. IR beats fixed chunking at equal compute:** success(IR@b) − success(k_b) > 0 with the 95% CI above 0, for b = 2
  and for b = 6 (both required).
- **G3. IR beats the newest transferable baselines:** success(IR@4) − success(FI@4) > 0 and
  success(IR@4) − success(RS@4) > 0, each with the 95% CI above 0 (each reported as its own sub-claim, G3a / G3b).
- **G4. No ensemble needed:** success(IR1@4) − success(random@4) > 0 with the 95% CI above 0.

**Statistics.** As before: differences paired by seed within each robot, averaged within robots and then over robots;
95% CIs from a bootstrap that resamples episodes within each robot.

**Reported, not claimed:** the success-vs-compute table and plot for every gate; act, reg, oracle, LBA, PE against
random and against fixed k at budgets 2 and 6; IR − act and IR − LBA at 2 and 6; IR − IR1; FI and RS against random;
IR@4 − k4 (already observed in the confirmation data, so not a claim); each new score's accuracy (step-stratified
AUROC) on the calibration plans for return-error and decision labels; achieved actions per plan.

## Procedure

A smoke job (1 robot, 2 episodes per arm, 2 calibration episodes) checks the pipeline first. Its numbers are not used.
Amendments are allowed only before the full jobs start, only for problems the smoke job finds, and are committed with
a reason.
