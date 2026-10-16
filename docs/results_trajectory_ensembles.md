# Trajectory-level ensembles (PETS / STEVE-style) vs the learned Bellman audit under compounding error

Question: one-step ensemble alarms (D, M: MOPO / MOBILE-style) go flat as imagination error compounds
(docs/results_compounding_final.md). Do trajectory-level ensembles, where each head imagines the whole rollout on its own,
do better? And is the reason ensembles miss errors a shared bias across heads (Proposition 2,
docs/theory_anchored_vs_ensemble.md)?

Setup: 15 ManiSkill3 robots (the main grounded agents with 5 dynamics heads), imagined rollouts of 24 and 40 steps from real
starts, the same hallucination labels as the main results (|imagined − real discounted reward sum|, calibrated threshold).
Signals: LBA (learned Bellman audit, one model), Aa / Ac (anchored / cumulative residuals), D / M (one-step ensembles),
Dt (PETS-style spread of independently rolled-out latents), Mt (STEVE-style spread of the heads' own imagined returns).
Tool: `python -m src.tools.compounding feat_*.tar.gz`. CIs are 2000-sample bootstraps over robots.

## 1. Trajectory ensembles DO get louder as error compounds

Growth of each signal on hallucinated windows, first 3 vs last 3 imagined steps (per-robot mean, robots with growth > 1):

| Signal | 24 steps | 40 steps |
| :-- | :-- | :-- |
| true error | large | ×37.5 |
| LBA | ×5.50 (15/15) | ×13.9 [6.8, 23.0] (15/15) |
| Dt | ×4.83 (15/15) | ×6.6 [5.6, 7.8] (15/15) |
| Mt | ×4.37 (15/15) | ×5.0 [2.4, 8.1] (13/15) |
| Aa | ×2.30 | ×2.6 [1.6, 4.0] (12/15) |
| D (one-step) | ×0.92 | ×0.89 [0.73, 1.04] (6/15) |
| M (one-step) | ×1.11 | ×0.76 [0.41, 1.21] (3/15) |

So the claim "ensembles go blind under compounding" holds only for **one-step** ensembles. Trajectory ensembles grow.

## 2. Ranking at the far end: the learned audit is best on average, decisively at 24 steps, less so at 40

Mean per-step AUROC over the last 3 imagined steps, LBA minus each ensemble (95% CI over robots):

| vs | 24 steps (15 robots) | 40 steps (15 robots) |
| :-- | :-- | :-- |
| D | +.135 [.085, .187], 15/15 | +.084 [+.006, +.152], 12/15 |
| M | +.170, 14/15 | +.130 [+.010, +.236], 12/15 |
| B | +.172, 14/15 | +.148 [+.036, +.263], 12/15 |
| Dt | +.149 [.099, .198], 14/15 | +.094 [−.026, +.200], 12/15 (sign test p ≈ .035) |
| Mt | +.105 [.038, .169], 13/15 | +.121 [−.014, +.236], 11/15 (sign test p ≈ .12) |

Mean late-step AUROC: 24 steps LBA .84, Mt .74, D .71, Dt .69. 40 steps LBA .81, D .72, Dt .71, Mt .69, Aa .65.

Reading: the learned audit (one model) out-ranks five-head trajectory ensembles at long range on average and on most
robots. At 24 steps the CIs exclude zero against every ensemble. At 40 steps they still exclude zero against the one-step
ensembles, but the CIs against Dt and Mt touch zero: a few robots favour the ensembles strongly. (The 12-robot interim at
40 steps excluded zero; the three rerun robots narrowed it.) Claim to make: "matches or beats trajectory ensembles at
long range with 1 model instead of 5", not "beats them significantly at every horizon".

Note the LBA AUROC stays flat (.81 → .81) from the first to the last steps at 40, while every other signal drops.

## 3. Shared bias: not supported

Heads rolled out independently from the real start (40 steps, mean over 15 robots):

| quantity | first 3 | last 3 | growth |
| :-- | :-- | :-- | :-- |
| latent error of the ensemble mean | 0.42 | 2.09 | ×4.9 |
| latent spread across heads | 0.22 | 1.76 | ×8.2 |
| return error of the ensemble mean | 0.07 | 2.52 | ×35 |
| return spread across heads | 0.02 | 1.35 | ×80 |

Error / spread **falls** with depth on 15/15 robots (latent 1.94 → 1.19; return 4.47 → 1.92), the same at 24 steps.
The heads diverge faster than their shared error grows, so Proposition 2's failure mode (members wrong together) is
possible in principle but **not observed** here. The error of the ensemble mean is still larger than the spread at the
end (14/15 latent, 15/15 return), i.e. ensembles remain over-confident in absolute terms, but they do not become more so.

## What this changes in the story

| Claim | Status |
| :-- | :-- |
| One-step ensemble alarms go flat under compounding (15 robots, 30 DMC tasks, 317M model) | holds |
| All ensembles go blind | **dropped**: trajectory ensembles grow |
| Shared bias explains ensemble failures | **dropped**: not observed |
| Learned audit ranks best at long range, 1 model vs 5 | holds on average; significant at 24 steps, directional (12/15, 11/15 robots) at 40 |
