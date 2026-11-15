# Pre-registration: is detection worth more when the robot commits to its imagination?

Written on 2026-10-20, before any result of this experiment existed. The commit that adds this file is the timestamp.
Every result is reported, pass or fail.

## Problem

In the run-time warning study (docs/monitor_results.md) even a perfect warning (oracle) raised success only from 0.580
to 0.648, and a random warning already gave 0.615. The robot replans after every action, so a hallucinated plan costs
at most one action before reality corrects it. Our hypothesis: **the value of detection is bounded by the cost of
acting on a hallucination, and that cost grows with how many actions the robot executes before it looks again.**
Executing several planned actions open-loop (action chunking) is common in practice because it saves planning
compute, and it is exactly where acting on a hallucination is costly.

**Question.** When the robot executes k planned actions before replanning, does the learned Bellman audit (LBA),
which judges the plan *before acting*, choose k well enough to improve success at the same planning compute?

## Design (code: `src/commit.py`, analysis: `src/tools/commit_report.py`)

- **Planner:** MPPI, 12-step horizon, standard scoring (the monitor study's H12). After each planning call the robot
  executes the first k actions of the chosen plan, then replans. MPPI's warm start is shifted by the executed k.
- **Gates** (k at most 8):
  - fixed k = 1, 2, 4, 8;
  - before acting, from the plan alone: commit to the longest prefix whose running-max per-step score stays below a
    threshold. Scores: LBA; Aa (label-free anchored residual); D (dynamics ensemble); M (MOBILE-style); oracle (the
    plan's true return error from executing it in a simulator copy, a ceiling);
  - while acting: PE, the prediction-error check, which executes until the encoded real state differs from the
    imagined one by more than a threshold (it sees reality; the before-acting gates do not);
  - random: continue to each next action with a fixed probability (control);
  - LBA+PE (combination, reported only): k from the LBA, and stop early on the PE check.
- **Matched compute:** each gate's threshold is set so that its mean commitment on calibration plans is 2 or 4 actions
  per plan (budgets 2 and 4), the same planning compute as fixed k = 2 or 4. Calibration: 10 closed-loop k = 1
  episodes per robot (seeds 6900–6909), every plan also executed in the simulator. Ties (e.g. saturated
  probabilities) pass with the probability that makes the match exact. Achieved actions per plan are reported.
- **LBA:** exactly as in the monitor study (`fit_lba`, the robot's 50 saved real episodes).
- **Arms:** 19 per robot, 100 evaluation episodes each, seeds 6000–6099 (the monitor study's seeds), the same for every
  arm: k1, k2, k4, k8, {oracle, LBA, Aa, D, M, PE, random} at budgets 2 and 4, LBA+PE@4.
- **Robots:** the 10 grounded robots of the monitor study except Peg s10 (success 0.05 leaves nothing to lose):
  Push s10, s40; StackCube s10, s20, s30, s40; YCB s10, s20, s30, s40.

## Claims (decided now)

- **C0. The setting makes hallucinations costly** (a precondition, not a claim about our method):
  success(k1) − success(k4) > 0 and success(oracle@4) − success(random@4) > 0, both with the 95% CI above 0.
  If C0 fails, C1–C3 are still reported, but detection has little to buy in this setting.
- **C1. The audit pays off under commitment:** success(LBA@4) − success(random@4) > 0, 95% CI above 0.
- **C2. Better than ensemble gates:** success(LBA@4) − success(X@4) has a positive mean for X = D and X = M.
- **C3. The value of detection grows with commitment:** [LBA − random]@4 − [LBA − random]@2 has a positive mean.

**Statistics.** As in the monitor study: differences paired by seed within each robot, averaged within robots and then
over robots; 95% CIs from a bootstrap that resamples episodes within each robot.

**Reported, not claimed:** the same comparisons at budget 2; Aa (label-free) against random; LBA against PE and LBA+PE
against PE (PE sees the real next state, so it may well be stronger; the question is whether the two are
complementary); LBA@4 against fixed k1, k2, k4; the share of the oracle's gain over random captured by the LBA;
each gate score's accuracy (step-stratified AUROC) on the calibration plans; achieved actions per plan.

## Procedure

A smoke job (1 robot, 2 episodes per arm) checks the pipeline first. Its numbers are not used. The pre-registration
may be amended only before the full jobs start, and only for problems found by the smoke job; amendments are
committed with a reason.

## Amendment 1 (2026-10-20, after the smoke job, before the full jobs)

**Problem found by the smoke job:** thresholds that match the budget exactly on the calibration plans did not transfer
to the evaluation episodes. Score levels shift between episodes, so at budget 4 the gates committed to 1.0 (D) to 2.2
(PE) actions per plan, while random committed to 4.6. Gates would then differ in planning compute, and a gate could
win just by replanning more often, which breaks the matched-compute design.

**Change:** a budget controller, the same for every score-based gate (oracle, LBA, Aa, D, M, PE, and the LBA in
LBA+PE). The calibration plans give a threshold for every commitment level 1.00, 1.05, …, 8.00. At run time a gate
uses the threshold of level = budget + 0.05 × Σ(budget − k) over its decisions so far in the arm. This integral
control drives the gate's running mean commitment to the budget. Plans cut short by the episode's end are not
counted. Fixed-k and random arms are unchanged (random already has the budget as its expected commitment). The claims,
arms, robots, seeds and statistics are unchanged. Smoke numbers are not used.
