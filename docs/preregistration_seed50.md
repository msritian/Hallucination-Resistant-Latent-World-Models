# Pre-registration: seed-50 confirmation of the learned Bellman audit

Written on 2026-10-02, before any seed-50 robot was trained. The commit that adds this file is the timestamp.
Nothing below may change after results arrive. Every result is reported, pass or fail.

## Question

Does the learned Bellman audit (LBA) still detect real hallucinations better than the published baselines on robots
that played no part in designing it, with every choice fixed in advance?

## Robots (new, never used before)

8 training runs, listed in `cluster/runs_seed50.txt`. There is one robot with our critic ("grounded", which has the
4 extra world models the ensemble baselines need) and one with the standard critic ("stock") for each of these tasks:

- PushCube-v1 (200k steps);
- StackCube-v1 (500k steps);
- PickSingleYCB-v1 (1M steps);
- PegInsertionSide-v1 (1M steps).

All runs use training seed 50 and the same training settings as the seed-40 robots.
A robot counts only if training finished. A failed run is reported as failed and is not rerun with other settings.

## Fixed method (code as of the commit adding this file)

- **Episodes:** 50 fresh episodes per robot, environment seeds 2000–2049. The first 25 are for calibration and the
  other 25 for testing. Imagination horizon is 12, beta = 1.0 (`run_preflight` defaults).
- **Labels (simulator, critic-free):** `return_error`, the difference between imagined and real discounted return.
  - Positive: above the 99th percentile of calibration step-0 error.
  - Negative: at or below the 90th percentile.
  - Sudden vs gradual: split by increments above the 99th percentile.
- **Ours (LBA):** `learned_audits` in `src/preflight/run_preflight.py`.
  - Features: `CRITIC_FEATURES` (A, Aa, Ac, Ab, Acb, P, B, E, At, Ao, A2, A3, A5, A8) plus the running max of each,
    v_hat, r_hat and step.
  - Model: `HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=0)`.
  - Fitted on the calibration episodes only.
- **Baselines, used as published:**
  - D: dynamics-ensemble disagreement;
  - M: MOBILE-style target spread;
  - B: critic disagreement;
  - E: ELVIS-style UCB.
- **Score:** step-stratified AUROC on the test episodes, for all, gradual and sudden hallucinations.

## Pass criteria (decided now)

Computed on the 4 new grounded robots, where every method exists:

1. **Primary:** the mean AUROC of ours is higher than each of D, M, B and E, for all hallucinations.
2. **Gradual and sudden:** the same comparison holds for gradual and for sudden hallucinations. This is reported
   separately; failing it on one case is reported as a partial pass.
3. **No significant loss:** on no robot is ours significantly worse than D or M. This uses a paired bootstrap over
   test episodes, 300 resamples, with a 95% interval entirely below 0.
4. **Bellman matters:** averaged over all 8 new robots, ours beats the "no Bellman" control (the same trees on
   v_hat, r_hat and step only) by at least 0.03 AUROC.

Secondary, with no pass or fail: ours vs B and E on the 4 standard-critic robots; and pooled results over all
15 grounded robots (11 earlier + 4 new).

## Commands (fixed now)

On the AP:

```
condor_submit train.sub queue_file=runs_seed50.txt
condor_submit confirm.sub queue_file=realcheck_s50.txt
```

On the Mac, after copying `results/real_seed_grounded_*_s50.tar.gz`:

```
.venv/bin/python -m src.tools.detection_tables <real_*_s50 tarballs>    # criteria 1, 2
.venv/bin/python -m src.tools.audit_ablations <real_*_s50 tarballs>     # criteria 3 (section 4), 4 (section 1)
```
