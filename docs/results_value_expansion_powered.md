# Value expansion, powered study: results (pre-registered in docs/preregistration_value_expansion_powered.md)

Tool: `python -m src.tools.ve_report <dir>`. 65 training runs (5 seeds, 80-84), all complete. Learning speed = mean
evaluation success at steps >= 50k; final = success over 100 episodes at the end of training.

## Summary

- **All pre-registered criteria fail (P1-P5).** Weighting imagined returns by the audit's trust does not make
  TD-MPC2 learn faster than the ensemble trust (D), than a constant trust, than no value expansion, or than the
  trajectory ensemble (Mt).
- The pilot's advantage for the audit over D did not replicate: lba − D = −0.051 [−0.118, +0.018] (A+B), audit
  higher on 3/10 seed pairs.
- Final success shows the same picture: no difference from D (+0.000) or Mt (−0.020); the largest gap (+0.059 vs no
  expansion) has a CI spanning zero.
- Reading: like the planner, warning-light and commitment studies, a more accurate detector does not translate into
  better control here. Learning is insensitive to which trust signal weights the imagined returns.
- Operational note: 26 runs crashed at start-up (staging file-count limit) and were rerun without the staging mirror;
  one run (D, pick, h5, s83) was evicted and resumed from its mirrored checkpoint at step 178,400.

## Full output
```

Cell A: stack h=5   mean over seeds (n runs)   speed | final
  none   (5)  0.136 | 0.532   seeds: 0.15 0.19 0.08 0.16 0.11
  const  (5)  0.132 | 0.510   seeds: 0.13 0.13 0.11 0.19 0.09
  D      (5)  0.141 | 0.516   seeds: 0.11 0.17 0.12 0.13 0.18
  Mt     (5)  0.131 | 0.542   seeds: 0.14 0.13 0.12 0.13 0.13
  lba    (5)  0.109 | 0.516   seeds: 0.15 0.08 0.13 0.09 0.09

Cell B: pick h=5   mean over seeds (n runs)   speed | final
  none   (5)  0.483 | 0.754   seeds: 0.67 0.51 0.63 0.50 0.10
  const  (5)  0.472 | 0.816   seeds: 0.60 0.46 0.52 0.68 0.09
  D      (5)  0.641 | 0.888   seeds: 0.71 0.63 0.71 0.69 0.46
  Mt     (5)  0.533 | 0.902   seeds: 0.71 0.66 0.41 0.61 0.27
  lba    (5)  0.572 | 0.888   seeds: 0.65 0.58 0.45 0.54 0.64

Cell C: stack h=10   mean over seeds (n runs)   speed | final
  const  (5)  0.137 | 0.524   seeds: 0.10 0.13 0.09 0.14 0.22
  D      (5)  0.117 | 0.508   seeds: 0.11 0.13 0.14 0.07 0.13
  lba    (5)  0.140 | 0.462   seeds: 0.18 0.10 0.14 0.10 0.17

Pre-registered criteria (learning speed; a verdict only when all pairs exist)
  P1 lba - D (A+B, speed): -0.051 [-0.118, +0.018], lba higher on 3/10 -> FAIL
  P2 lba - none (A+B, speed): +0.031 [-0.061, +0.163], lba higher on 5/10 -> FAIL
  P3 lba - const (A+B, speed): +0.038 [-0.057, +0.170], lba higher on 6/10 -> FAIL
  P5 lba - Mt (A+B, speed): +0.008 [-0.050, +0.097], lba higher on 4/10 -> FAIL
  P4 lba - D (C, speed): +0.023 [-0.007, +0.053], lba higher on 3/5 -> FAIL
     (compare with cell A: lba - D (A, speed): -0.033 [-0.076, +0.011], lba higher on 2/5)

Secondary: final success
  lba - D (A+B, final): +0.000 [-0.022, +0.022], lba higher on 5/10
  lba - none (A+B, final): +0.059 [-0.021, +0.177], lba higher on 6/10
  lba - const (A+B, final): +0.039 [-0.050, +0.163], lba higher on 4/10
  lba - Mt (A+B, final): -0.020 [-0.062, +0.022], lba higher on 3/10
  lba - D (C, final): -0.046 [-0.160, +0.036], lba higher on 2/5

Per cell (speed)
  lba - none (A, speed): -0.027 [-0.077, +0.019], lba higher on 2/5
  lba - const (A, speed): -0.023 [-0.067, +0.014], lba higher on 3/5
  lba - D (A, speed): -0.033 [-0.076, +0.011], lba higher on 2/5
  lba - Mt (A, speed): -0.022 [-0.045, +0.000], lba higher on 2/5
  lba - none (B, speed): +0.089 [-0.084, +0.332], lba higher on 3/5
  lba - const (B, speed): +0.099 [-0.077, +0.338], lba higher on 3/5
  lba - D (B, speed): -0.069 [-0.195, +0.065], lba higher on 1/5
  lba - Mt (B, speed): +0.039 [-0.073, +0.212], lba higher on 2/5
  lba - const (C, speed): +0.003 [-0.041, +0.053], lba higher on 2/5
  lba - D (C, speed): +0.023 [-0.007, +0.053], lba higher on 3/5
```
