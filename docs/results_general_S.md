# Results: generalisation study, Part S (stock TD-MPC2)

Pre-registration: docs/preregistration_general.md (commit 3ebdf6e). 10 stock robots, cluster 8541743, evaluation seeds
6300-6399, 100 episodes per arm. Analysis: `python -m src.tools.general_report S results/gs_*.tar.gz`. Every claim is
reported. Part N (new tasks) is still training and is reported separately.

**Inclusion rule:** 9 robots included; PickSingleYCB s50 excluded (collection success 0.06 < 0.20).

## Claims

| claim | result | verdict |
|---|---|---|
| S1 IR@4 − random@4 | +0.027 [−0.010, +0.063] | FAIL |
| S2 IR@4 − k4 | −0.013 [−0.048, +0.021] | FAIL |
| S2 IR@6 − k6 | −0.014 [−0.049, +0.018] | FAIL |

## Arms (mean over 9 robots)

| arm | k1 | k2 | k4 | k6 | random@4 | IR@2 | IR@4 | IR@6 | PE@4 |
|---|---|---|---|---|---|---|---|---|---|
| success | 0.609 | 0.546 | 0.463 | 0.380 | 0.423 | 0.530 | 0.450 | 0.366 | 0.451 |
| actions/plan | 1.00 | 2.00 | 3.85 | 5.56 | 3.77 | 2.03 | 3.97 | 5.76 | 3.94 |

Per robot:

| robot | k1 | k2 | k4 | k6 | random@4 | IR@2 | IR@4 | IR@6 | PE@4 |
|---|---|---|---|---|---|---|---|---|---|
| push s40 | 0.96 | 0.96 | 0.93 | 0.96 | 0.93 | 0.97 | 0.98 | 0.92 | 0.94 |
| push s50 | 0.93 | 0.92 | 0.94 | 0.88 | 0.89 | 0.92 | 0.93 | 0.92 | 0.86 |
| stack s20 | 0.70 | 0.54 | 0.36 | 0.32 | 0.40 | 0.54 | 0.33 | 0.30 | 0.46 |
| stack s30 | 0.62 | 0.58 | 0.43 | 0.25 | 0.37 | 0.56 | 0.43 | 0.28 | 0.42 |
| stack s40 | 0.71 | 0.49 | 0.40 | 0.16 | 0.36 | 0.53 | 0.36 | 0.23 | 0.41 |
| stack s50 | 0.70 | 0.63 | 0.52 | 0.28 | 0.39 | 0.58 | 0.43 | 0.24 | 0.38 |
| ycb s20 | 0.36 | 0.30 | 0.32 | 0.28 | 0.21 | 0.29 | 0.25 | 0.14 | 0.27 |
| ycb s30 | 0.29 | 0.27 | 0.16 | 0.16 | 0.13 | 0.21 | 0.24 | 0.12 | 0.20 |
| ycb s40 | 0.21 | 0.22 | 0.11 | 0.13 | 0.13 | 0.17 | 0.10 | 0.14 | 0.12 |

## Reported (not claimed)

IR@2 − k2 −0.016 [−0.053, +0.020]; PE@4 − random@4 +0.028 [−0.007, +0.063]; PE@4 − k4 −0.012; IR@4 − PE@4 −0.001;
k1 − k2 +0.063, k1 − k4 +0.146, k1 − k6 +0.229 (all CIs above 0).

## Reading

On stock TD-MPC2, committing costs a lot (k1 − k4 = +0.146, more headroom than on grounded robots), but neither IR nor
PE recovers any of it: both sit at the level of fixed chunking. IR's success on grounded robots therefore does not
transfer to the unmodified model.

The ensemble is not the missing piece: single-head IR (IR1) beat random by +0.080 [+0.050, +0.111] on grounded robots
(docs/results_frontier.md, G4). The remaining difference is grounding: the stock critic is trained on the world model's
imagined latents, the grounded critic on encoded real observations only (docs/02_how_it_works.md). IR evaluates the
critic on imagined states, and a critic trained to agree with the model's imagination cannot flag where imagination
goes wrong. This is a hypothesis, not a finding. The stock robots also differ from the grounded study robots in seeds and
training recipe, so the comparison is confounded until stock and grounded models are trained with the same recipe.
