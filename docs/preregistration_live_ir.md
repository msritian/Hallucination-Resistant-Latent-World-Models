# Pre-registration: live imagined regret (IR at the robot's real observation, while acting)

Written on 2026-11-16, before any live-IR arm has run. The commit that adds this file is the timestamp. Every result
is reported, pass or fail.

## Why

In the robust-regret study (docs/results_robust_regret.md, exploratory), IR evaluated at the encoded REAL states
(arm IRs, designed as an oracle) closed about 70% of the gap between IR and the regret oracle, and beat PE by +0.053
(budget 4) and +0.073 (budget 6) on the 12 frontier robots. Because the commitment rule is sequential (running maximum,
stop at the first failing step), IR at the robot's current observation, computed while acting, takes the same
decisions with no simulator. That comparison was post hoc, on the robots it was found on. This study tests it as a
real, simulator-free gate on robots it was not found on.

## Gate (src/robust_regret.live_ir, src/commit.py --study live)

Before executing plan action j (j ≥ 1), encode the current real observation z and compute
IRl_j = mean over critic heads h of max(0, Q_h(z, π(z)) − Q_h(z, a_j)); stop and replan at the first step whose running
maximum fails the threshold (same `passes` rule, same budget controller as every gate). No simulator, no dynamics
model. Thresholds: IRs scores on the closed-loop k = 1 calibration episodes (seeds 6900-6909, the protocol of every
earlier study; the simulator is used only there, as for every gate's thresholds).

Arms: k1, k2, k4, k6, random@4, IRl@2/4/6, PE@2/4/6 (arms that already exist in a robot's directory are reused:
same robot, same evaluation seeds).

## Confirmation set and claims

**Part N robots** (12 grounded robots on PickCube, PokeCube, PullCubeTool, LiftPegUpright, seeds 70-72; evaluation seeds
6300-6399; never used in the robust study). Each claim over the 12 robots:

- **L1. Live IR beats PE:** success(IRl@4) − success(PE@4) > 0, 95% CI above 0.
- **L2. Live IR beats PE at budget 6:** success(IRl@6) − success(PE@6) > 0, 95% CI above 0.
- **L3. Live IR beats plan-time IR:** success(IRl@4) − success(IR@4) > 0, 95% CI above 0.

Statistics as in every earlier study: paired by seed within robots, averaged within then over robots, bootstrap over
episodes within robots.

Note on power: Part N tasks are easy (k1 = 0.945), so effects are expected to be small; a failure here with a positive
point estimate would be reported as such, not reinterpreted.

## Reported, not claimed

- IRl@2 − k1 and IRl@2 − k2; IRl@b − k_b; IRl@4 − random@4; PE@2/6 − k2/k6.
- **Check set** (the 12 frontier robots of the robust study, seeds 6000-6099): IRl − IRs at budgets 4 and 6, which
  should be about 0 if the online gate reproduces the oracle (differences come from simulator non-determinism only),
  plus IRl@2 against k1, k2, IR@2, PE@2.
- **Stock set** (9 stock TD-MPC2 robots of Part S; PickSingleYCB s50 excluded by the Part S inclusion rule): the same
  comparisons as L1-L3. Live IR uses only the encoder and critic, so this asks whether a stock critic suffices when the
  state is real.

## Procedure

A smoke job (1 robot, 2 episodes per arm) checks the pipeline first. Smoke numbers are not used. Amendments only before
the full jobs start, only for problems the smoke finds, committed with a reason.

## Amendment 1 (2026-11-16, before any live-IR or AdaReP-style result was looked at)

Reason: a closely related method appeared in a literature check after the full jobs were submitted: AdaReP (Cheng et
al., arXiv 2606.23079, ICANN 2026) adapts when a world-model MPC replans, from the deviation of the real latent from
the cached rollout and a local sensitivity estimate. PE is its special case without adaptation, so it must be compared
directly. Three of the 12 Part N jobs had finished when this amendment was written; their outputs were not opened.

**AdaReP-style gate ADR** (src/adarep.py): score s = log d + d̄ / median(d) + L̄ / median(L̂), d = ||z_real − ẑ_j|| (main
head), L̂ = ||z_{t+1} − z_t|| / (||u_t|| + 1e-3), d̄ and L̄ means over the last W = 3 steps of the current plan; medians
from the calibration plans. This is AdaReP's rule "d > ε0·exp(−α_d d̄)·exp(−α_L L̄)" in log form, with ε0 set by the
budget controller like every gate's threshold, and α_d, α_L fixed by the median normalisation (the paper's TD-MPC2
values were not available to us). Arms ADR@2/4/6, run as a second job on each robot's live-IR archive (same seeds,
same calibration episodes), cluster/commit_adr.sub.

**Added claims (Part N robots):**
- **L4.** success(IRl@4) − success(ADR@4) > 0, 95% CI above 0.
- **L5.** success(IRl@6) − success(ADR@6) > 0, 95% CI above 0.

Reported: ADR@b − PE@b, ADR@2 − k1, IRl@2 − ADR@2, and the same on the check and stock sets.
