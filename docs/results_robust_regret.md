# Results: robust imagined regret (exploratory)

Code: src/robust_regret.py, `src/commit.py --study robust` (commit 1a44813). 12 frontier robots (grounded PushCube,
StackCube, PickSingleYCB; maturity seeds 60-62 and seed 50), cluster 8690619, evaluation seeds of every earlier arm,
100 episodes per arm, budgets 4 and 6. Analysis: `python -m src.tools.robust_report results/rr_{mat,seed}_*.tar.gz`
(the two October `rr_bk_*`, `rr_p4_*` archives belong to the older re-ranking study and are not part of this one).
Exploratory: nothing here was pre-registered. Smoke check: the r = 0 bound reproduces IR (max relative difference 1e-7).

## Success (mean over 12 robots)

| arm | k1 | k4 | IR@4 | reg@4 | act@4 | IRs@4 | RIRo@4 | RIR@4 | RIRt@4 | RIRc@4 |
|---|---|---|---|---|---|---|---|---|---|---|
| success | 0.743 | 0.635 | 0.667 | 0.722 | 0.679 | 0.707 | 0.671 | 0.643 | 0.652 | 0.659 |

| arm | k6 | IR@6 | reg@6 | act@6 | IRs@6 | RIRo@6 | RIR@6 | RIRt@6 | RIRc@6 |
|---|---|---|---|---|---|---|---|---|---|
| success | 0.542 | 0.598 | 0.647 | 0.602 | 0.640 | 0.581 | 0.593 | 0.568 | 0.588 |

## Differences (bootstrap 95% CI over episodes within robots)

| comparison | budget 4 | budget 6 |
|---|---|---|
| IRs − IR (state part of the gap) | +0.039 [+0.012, +0.065] | +0.043 [+0.013, +0.072] |
| reg − IRs (action-proposal part) | +0.015 [−0.011, +0.041] | +0.007 [−0.023, +0.034] |
| RIRo − IR (true radius, ceiling of the robust idea) | +0.003 [−0.025, +0.032] | −0.017 [−0.047, +0.013] |
| RIR − IR (detector radius) | −0.024 [−0.052, +0.005] | −0.005 [−0.035, +0.025] |
| RIRt − IR (ensemble radius) | −0.016 [−0.042, +0.011] | −0.029 [−0.058, +0.000] |
| RIRc − IR (horizon-only radius) | −0.008 [−0.037, +0.020] | −0.009 [−0.039, +0.021] |
| RIR − RIRt | −0.008 [−0.037, +0.020] | +0.024 [−0.005, +0.053] |

Offline (calibration plans, ranking the regret oracle, AUROC top quarter vs bottom half): IR 0.818, IRs 0.884,
RIRo 0.793, RIR 0.779, RIRt 0.778, RIRc 0.777.

## Reading

1. **Robust regret fails.** Even with the true radius (RIRo), the first-order worst-case term does not improve on IR,
   closed loop or offline; with the detector's radius it is slightly worse. Adding slope × radius inflates the score
   of every step with a steep critic, whether or not the real state lies in the bad direction. The detector-sized
   radius therefore does not bring the detector back into control.
2. **The gap is the state, not the action proposal.** IR evaluated at the encoded real states (IRs) closes about 70%
   of the IR → regret-oracle gap (+0.039 of +0.054 at budget 4); the remainder (reg − IRs, the fresh plan's action vs
   the policy's) is not significant.
3. **IRs is not really an oracle.** The commitment rule is sequential (running maximum, stop at the first failing
   step), and the real state at step j is exactly the state the robot reaches by executing the plan. So IR at the
   robot's current observation, computed while acting (like PE), gives the same decisions with no simulator
   (deterministic CPU simulation assumed). Post hoc comparisons for this "live IR": vs PE +0.053 [+0.025, +0.080]
   (11/12 robots) at budget 4 and +0.073 [+0.043, +0.102] at 6; vs k4 +0.072; vs act oracle +0.028 [−0.000, +0.054];
   vs k1 −0.037 [−0.061, −0.013]. These are post hoc and need a pre-registered test with a genuinely online
   implementation on fresh robots.
