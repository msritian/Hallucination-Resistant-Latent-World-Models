# Pre-registration: does the IR gate save real time? (wall-clock study)

Written on 2026-11-12, before any result of this experiment existed. The commit that adds this file is the timestamp.
Every result is reported, pass or fail.

## Why

The frontier study (docs/results_frontier.md) showed that gating on imagined regret (IR) beats fixed chunking at
equal numbers of planning calls. It did not show that the robot runs faster. In those jobs every gated arm computed
all the gates' scores for every plan (LBA trees on the CPU, RS, FI, ...), about 65 ms per plan, more than a planning
call (about 48 ms), so IR@2 ran slower than replanning every step (0.88x). That is a cost of the research harness,
not of IR. IR itself needs, per plan, one encoding, the M dynamics heads' rollouts of the plan (L steps) and two
critic evaluations on M x L states, far less than an MPPI planning call. Also, the old arms of one robot ran on
different GPUs (confirmation vs frontier jobs), so their times are not comparable.

## Design (code: `src/commit.py --study timing`, cluster/commit_timing.sh)

- **Lean scorer** `ir_only`: computes IR alone, with exactly the operations of the frontier study's IR. Each job
  first checks on 5 plans that it matches the old scorer (max difference < 1e-5) and stops otherwise.
- **Arms:** k1, k2, k4, k6, IR@2, IR@4, IR@6 (lean). All arms of a robot run in one job on one GPU, one after another.
- **Thresholds and L:** reused unchanged from each robot's earlier jobs (thresholds.json, lba.pkl). No recalibration.
- **Robots:** the 12 of the confirmation and frontier studies.
- **Seeds:** 100 evaluation episodes per arm, seeds 6100-6199, never used in any commitment study.
- **Timing:** per plan, planning time and gate time, measured with CUDA synchronisation (k arms get the same
  synchronisation). Per episode, the total control-loop time per step (planning + gate + simulator).

## Claims (decided now)

- **W1. The gate is cheap:** mean over robots of (total gate time / total planning time) over the IR arms < 0.15.
- **W2. IR saves real time:** time per step of k1 / time per step of IR@2 > 1, 95% CI above 1.
- **W3. The success gain replicates on fresh seeds:** success(IR@b) − success(k_b) > 0 with the 95% CI above 0,
  for b = 2 and for b = 6.
- **W4. The gain is not bought with hidden compute:** for b = 2 and 6, mean over robots of
  time per step of IR@b / time per step of k_b < 1.15.

**Statistics.** As in the earlier studies: success differences paired by seed within each robot, averaged within
robots and then over robots; 95% CIs from a bootstrap that resamples episodes within each robot. Time ratios: per
robot, the ratio of mean times per step; averaged over robots; CI from the same bootstrap.

**Reported, not claimed:** IR@4 vs k4; k1 − IR@b; speedups of every arm over k1; achieved actions per plan;
planning and gate time per plan in ms; GPU models.

## Procedure

A smoke job (1 robot, 2 episodes per arm) checks the pipeline first. Its numbers are not used. Amendments are allowed
only before the full jobs start, only for problems the smoke job finds, and are committed with a reason.
