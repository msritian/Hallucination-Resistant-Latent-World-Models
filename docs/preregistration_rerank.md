# Pre-registration (pilot): correcting over-optimistic plans with the Bellman audit (Idea 1)

Written 2026-10-06 before any result. Pilot = 2 robots: StackCube s10 and YCB s10 (grounded; H12 standard success
0.68 and 0.43, so room to improve). Code: `src/rerank.py`. 50 evaluation episodes per rule (seeds 7000+), 20
calibration episodes (seeds 7900+; correction fitted on the first 10, offline quality measured on the other 10,
then refitted on all 20 for the closed loop). At every step MPPI (H=12, standard score) runs, then one of its top-16
candidates is executed according to the rule: default, imagined, corrected (Idea 1), penD/penM (ensemble penalty,
lambda tuned on calibration for the baseline's benefit), oracle (true score from the simulator), random.

Go/no-go (decided now):
- G1 (is there room?): success(oracle) - success(imagined) >= 0.10, averaged over the 2 robots. If not, Idea 1 cannot
  matter in this setting -> move the test to an offline-trained (low-data) robot.
- G2 (does Idea 1 work?): offline held-out regret corrected < imagined and < best ensemble penalty on both robots, AND
  closed-loop success(corrected) > success(imagined) and >= success(best ensemble penalty), averaged over robots.
- If G1 and G2 pass: scale to all tasks/robots with a full pre-registration. Pilot numbers are reported either way.

Amendment (2026-10-06, before the pilot started): added a reference arm "H3" (standard 3-step planning, the TD-MPC2
setting). Extra condition for usefulness: success(corrected, H12) must exceed success(H3); otherwise Idea 1 only repairs
a planner that is worse than TD-MPC2's own.

## Round 2 (2026-10-07, after the pilot; exploratory, decided before these runs)
Pilot: YCB oracle 0.56 vs H3 0.36 (room exists), but no rule captured it. New rules, fitted on calibration only:
`relvalue` (trees predict each candidate's true value relative to the other candidates of the same decision, from
audit features + imagined score) and `valuepred` (true value directly). Runs: YCB H12, YCB H3, Peg H3 (H3 = TD-MPC2's
own horizon: measures the room for standard TD-MPC2). Promising if, on held-out calibration decisions, a learned rule
cuts regret clearly below imagined AND the ensemble penalties, and in closed loop beats H3 by >= 0.05.
