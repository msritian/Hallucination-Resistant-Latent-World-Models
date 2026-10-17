# Pre-registration: detector-guided training (train more where imagination fails)

Written on 2026-10-05, before any result of this experiment existed. The commit adding this file is the timestamp.
Every result is reported, pass or fail.

## Why

Using the detector to *choose between plans* failed in three pre-registered tests (warning light, audit-weighted λ,
candidate dropping). Dropping the flagged candidates was even worse than dropping at random (−0.038). The plans the
detector flags are the useful, contact-heavy ones, where the world model is weakest. So we test the opposite use:
**train the model more on exactly that experience**, so that the planner gets a better model where it matters.

## Method (code: `src/training/audit_replay.py`, `src/train.py --replay`)

- **Base:** standard TD-MPC2 with the 4 extra dynamics heads, which are trained on detached inputs and never shape
  the encoder. The run type is `stock_aux`, the same for every arm, so the ensemble control is available.
- **Rescoring:** every 50k environment steps, the current model re-scores every stored 12-step window of real
  experience:
  - **lba:** max over the window of p_t from the learned Bellman audit, refitted on the newest 50 episodes with the
    same recipe as the detection results;
  - **D:** max over the window of dynamics-ensemble disagreement (published-style control).

  The 30% highest-scoring windows form the priority set.
- **Batches:** half of each training batch is drawn from random 3-step slices inside priority windows. The other
  half is drawn uniformly from the replay buffer, exactly as TD-MPC2 normally samples.
- **Arms:** uniform (standard TD-MPC2), lba and D.
- **Tasks and seeds:**
  - StackCube (500k steps), PickSingleYCB (1M) and PegInsertionSide (1M);
  - training seeds 60 and 61 (new);
  - 3 × 2 × 3 = 18 runs (`cluster/runs_replay.txt`).
- **Final evaluation:** 100 episodes with TD-MPC2's own planner, on evaluation seeds 1000–1099 (the same for every
  run). We record per-episode success and the model's imagination error on those episodes: the mean |imagined −
  real| discounted return over 11 steps, which is the simulator answer key of the detection results.

## Claims (decided now)

- **T1 (main).** Final success, lba − uniform > 0, pooled over the 6 task-seed pairs. Paired by evaluation seed;
  95% CI from a bootstrap over episodes within each pair. Passes if the CI is above 0.
- **T2.** Final success, lba − D > 0 (same test): the audit is a better guide than ensemble disagreement.
- **T3.** Imagination error on the final evaluation episodes, lba < uniform, on the mean over the 6 pairs:
  training where the audit points reduces hallucination.

**Reported, not claimed:**
- per task and per seed;
- learning curves (evaluations every 50k steps, 10 episodes);
- imagination error of D.

**Limitation stated in advance:** 2 training seeds per task. The CIs cover evaluation-episode noise, not
training-seed variance, so the per-seed results are shown alongside them.

## Procedure

1. **Smoke run** (`runs_replay_smoke.txt`: PushCube, 6k steps, rescoring every 2k). It checks that every arm trains,
   rescores, evaluates and writes `final_eval.json`. It gives no results.
2. **Full run.** There are no code changes after the smoke run unless it fails; any change is logged here first.
3. **Analysis:** `python -m src.tools.replay_report results/rep_*.tar.gz`.
