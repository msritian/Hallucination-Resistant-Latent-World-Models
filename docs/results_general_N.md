# Results: generalisation study, Part N (new tasks)

Pre-registration: docs/preregistration_general.md (commit 3ebdf6e). 12 grounded robots on 4 tasks never used in the commitment work
(PickCube, PokeCube, PullCubeTool, LiftPegUpright; seeds 70-72), all trained to final-eval success 1.00. Cluster 8687304,
evaluation seeds 6300-6399, 100 episodes per arm. Analysis: `python -m src.tools.general_report N results/gn_*.tar.gz`.
Every claim is reported.

**Inclusion rule:** all 12 robots included (no collection success below 0.20).

## Claims

| claim | result | verdict |
|---|---|---|
| N1 IR@4 − random@4 | +0.042 [+0.022, +0.061] | PASS |
| N2 IR@4 − k4 | +0.032 [+0.013, +0.052] | PASS |
| N2 IR@6 − k6 | +0.037 [+0.015, +0.061] | PASS |

## Arms (mean over 12 robots)

| arm | k1 | k2 | k4 | k6 | random@4 | IR@2 | IR@4 | IR@6 | PE@4 |
|---|---|---|---|---|---|---|---|---|---|
| success | 0.945 | 0.942 | 0.912 | 0.863 | 0.902 | 0.970 | 0.944 | 0.901 | 0.950 |
| actions/plan | 1.00 | 2.00 | 3.88 | 5.64 | 3.81 | 2.01 | 3.96 | 5.80 | 3.97 |

Per robot:

| robot | k1 | k2 | k4 | k6 | random@4 | IR@2 | IR@4 | IR@6 | PE@4 |
|---|---|---|---|---|---|---|---|---|---|
| lift s70 | 0.95 | 0.97 | 0.96 | 0.91 | 0.93 | 1.00 | 0.97 | 0.98 | 0.96 |
| lift s71 | 0.97 | 0.95 | 0.95 | 0.84 | 0.94 | 0.98 | 0.96 | 0.87 | 0.94 |
| lift s72 | 0.84 | 0.77 | 0.68 | 0.63 | 0.70 | 0.91 | 0.83 | 0.70 | 0.85 |
| pick s70 | 0.95 | 0.97 | 0.94 | 0.90 | 0.89 | 0.97 | 0.93 | 0.92 | 0.98 |
| pick s71 | 0.98 | 0.97 | 0.91 | 0.90 | 0.91 | 1.00 | 0.97 | 0.93 | 0.98 |
| pick s72 | 0.91 | 0.93 | 0.91 | 0.84 | 0.93 | 0.98 | 0.94 | 0.94 | 0.97 |
| poke s70 | 0.90 | 0.90 | 0.85 | 0.73 | 0.85 | 0.90 | 0.90 | 0.85 | 0.90 |
| poke s71 | 0.97 | 0.98 | 0.96 | 0.95 | 0.95 | 0.99 | 0.97 | 0.93 | 0.98 |
| poke s72 | 0.95 | 0.96 | 0.94 | 0.93 | 0.92 | 0.98 | 0.96 | 0.97 | 0.95 |
| pull s70 | 0.96 | 0.94 | 0.92 | 0.88 | 0.98 | 0.96 | 0.96 | 0.87 | 0.94 |
| pull s71 | 0.98 | 0.99 | 0.96 | 0.90 | 0.88 | 0.98 | 0.98 | 0.90 | 0.96 |
| pull s72 | 0.98 | 0.98 | 0.96 | 0.95 | 0.95 | 0.99 | 0.96 | 0.95 | 0.99 |

## Reported (not claimed)

IR@2 − k2 +0.028 [+0.012, +0.043]; PE@4 − random@4 +0.048 [+0.027, +0.068]; PE@4 − k4 +0.038 [+0.020, +0.058];
IR@4 − PE@4 −0.006 [−0.023, +0.011]; k1 − k2 +0.003 [−0.014, +0.019]; k1 − k4 +0.033 [+0.014, +0.053];
k1 − k6 +0.082 [+0.059, +0.103].

Exploratory (not in the pre-registration, computed after seeing the table): IR@2 − k1 +0.025 [+0.009, +0.041],
9 robots better and 0 worse; IR@4 − k1 −0.001 [−0.018, +0.016]; PE@4 − k1 +0.005 [−0.012, +0.022].

## Reading

IR generalises to four new grounded tasks: it beats random gating and fixed chunking at both budgets. As on the
original tasks, it ties the prediction-error oracle PE, so the gain comes from gating at all, not from IR in particular.
The headroom is small here (k1 = 0.945), which is why the effects are about 0.03-0.04.

New, and exploratory: on these tasks, committing to 2 actions under IR (IR@2) does better than replanning every step
(+0.025, no robot worse), and IR@4 equals replanning at a quarter of the planning calls. On the frontier robots, IR at
budget 2 was below k1 (0.727 vs 0.743, docs/results_frontier.md). The difference may come from task difficulty. One
possible explanation is that a replanning MPPI planner re-samples every step and jitters, while a gate holds on to a
plan only where it is still good. This needs a pre-registered test before it can be claimed.

Together with Part S, the result supports the grounding hypothesis. IR works on grounded robots on new tasks and fails on
stock TD-MPC2. The confound noted in Part S (different seeds and training recipe) still holds. The grounding study
(cluster 8681982) trains stock and grounded models with the same recipe and settles this.
