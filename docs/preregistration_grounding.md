# Pre-registration: is grounding what makes IR work? (matched 2x2)

Written on 2026-11-13, after Part S of the generalisation study failed (docs/results_general_S.md) and before any model
of this study was trained. The commit that adds this file is the timestamp. Every result is reported, pass or fail.

## Why

IR beat fixed chunking on grounded robots (wall-clock studies 1 and 2, frontier study) but not on stock TD-MPC2
(Part S: IR@4 − k4 = −0.013). Two things differ: (1) grounding: the stock critic and policy train on the world
model's imagined latents, the grounded ones on encoded real observations only; (2) the grounded robots have 4 extra
dynamics heads. The stock robots of Part S also used other seeds and another training recipe. Hypothesis: grounding is
what makes IR work, because IR asks the critic to judge imagined states, and a critic trained on imagination agrees with
the model's mistakes. (Single-head IR1 already worked on grounded robots, frontier study G4, so the ensemble is not
needed at test time, but the extra heads also change training.)

## Design

Four variants, identical recipe (the maturity recipe of cluster/mat_train_runs.txt: PushCube 200k, StackCube 500k,
PickSingleYCB 1M steps), identical seeds 60, 61, 62, on the three tasks: 36 robots.

| variant (`--run`) | critic trained on | extra dynamics heads | models |
|---|---|---|---|
| grounded | real latents | 4 | existing mat_grounded_* (trained before this study) |
| grounded_noaux | real latents | 0 | new mat_noaux_* |
| stock_aux | imagined latents | 4 | new mat_stockaux_* |
| stock | imagined latents | 0 | new mat_stock_* |

Each robot is evaluated with the general-study protocol unchanged (src/commit.py --study general,
cluster/commit_general.sh): 50 collection episodes fit the LBA, 10 calibration episodes give thresholds, then arms k1,
k2, k4, k6, random@4, IR@2, IR@4, IR@6, PE@4, 100 episodes each, seeds 6300-6399. IR scorer: batched (ir_fast).
Inclusion rule as before: collection success ≥ 0.20; excluded robots are listed.

## Claims (decided now)

Per robot, gain G = success(IR@4) − success(k4), paired by evaluation seed. Group mean = mean of G over the group's
included robots; 95% CIs by bootstrap resampling episodes within each robot (5000 resamples).

- **H1. Grounding matters:** mean G over the two real-latent variants (grounded, grounded_noaux) − mean G over the two
  imagined-latent variants (stock, stock_aux) > 0, CI above 0.
- **H2. Grounding works without the ensemble:** mean G over grounded_noaux > 0, CI above 0.
- **H3. Grounding is needed:** mean G over stock (recipe-matched) < mean G over grounded, i.e.
  G(grounded) − G(stock) > 0, CI above 0.

**Reported, not claimed:** the same with b = 6 (IR@6 − k6) and with IR@4 − random@4; the ensemble main effect
(mean G with 4 extra heads − without); every variant's arm table; k1 − k_b per variant (headroom); PE@4 per variant;
per-task results.

## Procedure

A training smoke (the 3 new variants, PushCube, 20k steps) checks the code bundle first. Then 27 training jobs
(cluster/grounding_train_runs.txt) and 36 evaluation jobs (cluster/grounding_general_runs.txt; the 9 grounded robots
can start immediately). Analysis: `python -m src.tools.grounding_report results/ga_*.tar.gz`. Amendments only before
results exist, committed with a reason.
