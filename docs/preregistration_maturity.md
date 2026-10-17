# Pre-registration: does detection get better as the same robot trains longer? (training-maturity study)

Written 2026-10-16, before any of these robots was trained.

## Question

A reviewer's concern: under-trained world models make bigger, more obvious errors, so a detector could look good only
because the agents are immature. Across robots we already see the opposite (docs/results_difficulty.md: weak agents'
hallucinations are closer to the bar, and detection is slightly worse on them), but there "weak" is confounded with the
task (YCB/Peg are the weak ones). Here the task is held fixed and only the amount of training changes.

Prediction (from Proposition 1, Aa = |C − I|): as training goes on, the grounded critic gets more accurate, the critic
floor |C| shrinks relative to the values, and detection improves, even though the remaining errors get subtler.

## Design (fixed)

- Robots: grounded TD-MPC2, new seeds 60, 61, 62 on PushCube (200k steps), StackCube (500k) and PickSingleYCB (1M):
  9 robots, the same training settings as the seed-50 robots (cluster/mat_train_runs.txt).
- Snapshots: 10%, 25%, 50%, 75% and 100% of training (`--snapshots`; 100% = final_model.pt).
- Per snapshot, exactly the main protocol (run_preflight --dump_features): 50 new episodes with that snapshot's own
  planner (seeds 2000–2049), detector fitted on the first 25, scored on the last 25, H = 12, labels from the simulator
  with that snapshot's own thresholds. The dump also records the Proposition 1 split (C, I) per step.
- Analysis: `python -m src.tools.maturity results/mfeat_*.tar.gz`.

## Criteria (EARLY = 25%, LATE = 100%; per robot Δ = AUROC_LBA(LATE) − AUROC_LBA(EARLY); 95% bootstrap CI over robots)

| | Criterion | Pass if |
| :-- | :-- | :-- |
| M1 | Detection improves with training | mean Δ > 0 and the CI lower bound > 0 |
| M2 | Not inflated by immature models (the reviewer's concern) | CI lower bound of mean Δ > −0.03 |
| M3 | Mechanism: the critic floor shrinks | median \|C\| / (P95 − P5 of the critic's values) is lower at LATE than at EARLY on ≥ 7 of 9 robots |
| M4 | Sanity: the agents do improve | reported: mean success(LATE) − success(EARLY) with CI |

Reported, not criteria: all five snapshots per robot, per task; the anchored residual and ensemble D on the same
snapshots; the signal-to-floor ratio median |I| on hallucinated steps / median |C|; Spearman(success, AUROC) over all
(robot, snapshot) points.

## Rules

- A snapshot with fewer than 20 hallucinated or 20 correct test steps is excluded and reported as such. A robot without
  both the 25% and 100% snapshots is excluded from M1–M3. A failed job is rerun once with identical settings.
- The 10% snapshot is reported only (very early models may be degenerate).

## How each outcome will be reported

| Outcome | Reading |
| :-- | :-- |
| M1 pass | Detection improves as the same robot trains: immature agents make detection harder, not easier |
| M1 fail, M2 pass | Detection is stable across training: no sign that immature agents inflate the results |
| M2 fail | The concern holds: detection is better on immature agents; reported as a limitation |
| M3 pass / fail | Whether the improvement goes with a more accurate critic, as Proposition 1 predicts |
