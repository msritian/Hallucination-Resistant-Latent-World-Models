# Pre-registration: the learned Bellman audit inside the planner

Written on 2026-10-04, before any result of this experiment existed. The commit adding this file is the timestamp.
Every result is reported, pass or fail.

## Why this experiment

The warning-light test (`docs/monitor_results.md`) showed two things:

- The audit is accurate on the plans the planner acts on (AUROC 0.81).
- Checking only the *chosen* plan and switching to a 3-step plan was no better than switching at random.

Even a perfect check of the chosen plan helped only +0.07, because MPC replans every step. Here the audit acts
**inside the planner's search**, on every candidate and every imagined step. That is where over-optimistic
(hallucinated) plans win the competition between candidates.

## Planners (code: `src/planning/detector_planner.py`, `src/detector_planning.py`)

- **Base:** MPPI with the λ-return score, our best earlier planner (constant λ = 0.8, horizon 12).
  G_t = r̂_t + γ((1 − λ_t) V(ẑ_{t+1}) + λ_t G_{t+1}).
- **Audit:** the same learned Bellman audit as the detection results (same features, labels and trees; fitted on
  each robot's 50 saved real episodes). It runs on the GPU as an exact copy of the trees; every job checks the copy
  against sklearn before starting. It gives p_t for every candidate and imagined step.
- **Use 1, audit-weighted λ:** λ_t = 0.8 · (1 − p_t). Where a candidate's imagination is likely hallucinated, its
  score stops using the imagined future and uses the critic's value of the last trusted state instead.
- **Use 2, dropping candidates:** in every MPPI iteration, remove the 25% of candidates with the highest max_t p_t
  before the elites are chosen.
- **Controls:**
  - the same λ profiles, randomly reassigned across candidates (*shuffled*);
  - removing a random 25% of candidates.

## Arms

100 episodes each, seeds 6000–6099 (the same as the warning-light run), on the same 11 robots trained with our critic.

| Arm | Role |
| :-- | :-- |
| L08_H12 | base (constant λ 0.8, H = 12) |
| LBAlam_H12 | Use 1 |
| LBAlam_H12_shuffled | control for Use 1 |
| L08_H12_drop25 | Use 2 |
| L08_H12_dropRandom25 | control for Use 2 |
| LBAlam_pure_H12 (λ_t = 1 − p_t) | exploratory |
| L08_H24, LBAlam_H24 (audit fitted on 24-step windows) | exploratory: longer horizon |
| L08_H3 | exploratory: short horizon, same score |

## Claims (decided now)

- **P1.** LBAlam_H12 − L08_H12 > 0 (95% CI above 0): the audit improves our best planner.
- **P2.** LBAlam_H12 − LBAlam_H12_shuffled > 0 (95% CI above 0): the improvement comes from *which* candidates are
  flagged, not from lowering λ on average.
- **P3.** L08_H12_drop25 − L08_H12 > 0 (95% CI above 0): dropping suspicious candidates helps.
- **P4.** L08_H12_drop25 − L08_H12_dropRandom25 > 0 (95% CI above 0): the audit chooses which to drop better than
  chance.

**Statistics:** differences are paired by seed within each robot, averaged within robots and then over robots.
The 95% CI comes from a bootstrap that resamples episodes within each robot.

**Exploratory:**
- the H24 arms and the pure version;
- comparisons with standard H3 and H12 from the warning-light run (same robots and seeds), via
  `detplan_report --monitor`.

## Procedure

1. **Smoke job** (`detplan_smoke.txt`: one robot per task, 2 episodes per arm). It checks that the GPU trees match
   sklearn on the cluster, that every arm runs, and how long it takes. It gives no results.
2. **Full run** (`detplan_runs.txt`). There are no code changes after the smoke job unless it fails; any change is
   logged here first.
3. **Analysis:**
   `python -m src.tools.detplan_report results/plan_*.tar.gz --monitor results/mon_*.tar.gz`.
