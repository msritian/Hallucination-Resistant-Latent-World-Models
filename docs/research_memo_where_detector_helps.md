# Where can the Bellman audit add value? (research memo, 2026-10-08)

## Evidence so far
| Use | Outcome | Reason |
| :-- | :-- | :-- |
| Detection of hallucinated imagined steps | best (0.82 vs <=0.69), confirmed on new robots | judges imagination before reality, no extra models |
| Choosing among MPC candidates (5 forms) | no gain | candidates near-identical; replanning absorbs errors; flagged = useful contact plans |
| Test-time shift detection | prediction error better | the real next observation is available |
| Imagined policy evaluation | no better than ensembles | long rollouts of unseen policies, outside the audit's training |
| Horde (per-signal, reward-free) | reward-free ~ ensembles; no localisation gain | same underlying latent error seen through more windows; error concentrated in one signal |

## Principle
The audit is a situation-level reliability signal (accurate across situations), not a fine-grained ranker within one
situation. It adds value only where (a) ground truth is unavailable at decision time, (b) acting on situation-level
reliability changes the outcome, and (c) no cheaper/better signal exists. Acting-time uses in a replanning MPC fail (b);
learning from imagined data satisfies (a)-(c), and is where the literature reports large gains from uncertainty signals
(MOPO, MOReL, MOBILE offline; MBPO and adaptive-horizon methods online).

## Ranked directions
1. Offline policy learning with audit penalties on imagined value targets (MOBILE-style, ensemble-free).
2. Adaptive imagination length during learning (rollouts from real states with the agent's own policy).
3. Rerun detector-guided replay (bug fixed).
4. Bellman residual as a training loss (self-consistency).
