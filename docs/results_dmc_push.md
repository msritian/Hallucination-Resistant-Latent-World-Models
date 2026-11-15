# DMC disturbance study: results

Pre-registration: docs/preregistration_dmc_push.md (commit 33a0cbe). 5 tasks, push strength 0.5, budget 6, 30 episodes
per arm, seeds 8100-8129, all 5 jobs exit 0. Analysis: `python -m src.tools.dmc_push_report results/dpg_*.tar.gz`.

```
tasks: 5

task                      k1        k6  random@6      IR@6      PE@6     BND@6     reg@6
finger-turn-hard       0.981     0.909     0.882     0.953     0.946     0.950     0.980
hopper-stand           0.447     0.374     0.323     0.324     0.365     0.311     0.364
pendulum-swingup       0.642     0.544     0.524     0.534     0.615     0.615     0.612
walker-run             0.493     0.401     0.374     0.362     0.402     0.386     0.428
walker-walk            0.964     0.900     0.884     0.902     0.932     0.918     0.940
mean                   0.705     0.626     0.597     0.615     0.652     0.636     0.665
actions/plan            1.00      5.95      6.02      5.98      5.97      5.97      5.98

claims:
P1 k1 - k6                   +0.080 [+0.061, +0.102]  PASS
P2 BND@6 - IR@6              +0.021 [+0.000, +0.041]  PASS
P3 BND@6 - PE@6              -0.016 [-0.037, +0.005]  FAIL
P4 BND@6 - k6                +0.010 [-0.015, +0.035]  FAIL

reported:
IR@6 - k6                    -0.011 [-0.033, +0.014]  
IR@6 - random@6              +0.018 [-0.010, +0.047]  
PE@6 - k6                    +0.026 [+0.001, +0.051]  
PE@6 - random@6              +0.055 [+0.033, +0.078]  
reg@6 - k6                   +0.039 [+0.019, +0.063]  
reg@6 - random@6             +0.067 [+0.042, +0.095]  
random@6 - k6                -0.028 [-0.056, -0.002]  
random@6 - random@6          +0.000 [+0.000, +0.000]  
BND@6 - k6                   +0.010 [-0.014, +0.036]  
BND@6 - random@6             +0.039 [+0.010, +0.069]  

bound weights and calibration AUROC (reg top quarter vs bottom half, steps with reg > 0 or not):
  finger-turn-hard   a=0.1099 b=0.1243  IR 0.752  PE 0.494  BND 0.748
  hopper-stand       a=0.4635 b=0.0784  IR 0.765  PE 0.683  BND 0.845
  pendulum-swingup   a=0 b=0.2627  IR 0.838  PE 0.591  BND 0.591
  walker-run         a=0.8353 b=0.2119  IR 0.795  PE 0.758  BND 0.879
  walker-walk        a=0.6334 b=0.3154  IR 0.834  PE 0.745  BND 0.901
```

## Claims

| Claim | Result | Verdict |
| :-- | :-- | :-- |
| P1 k1 − k6 > 0 | +0.080 [+0.061, +0.102] | PASS |
| P2 BND − IR > 0 | +0.021 [+0.000, +0.041] | PASS (lower bound at 0 to three decimals) |
| P3 BND − PE > 0 | −0.016 [−0.037, +0.005] | FAIL |
| P4 BND − k6 > 0 | +0.010 [−0.015, +0.035] | FAIL |

## What it means

- Under pushes, committing costs 0.080 (P1), and the while-acting signal PE is the best label-free gate
  (+0.026 over k6, +0.055 over random). The decision oracle reg reaches +0.039 over k6.
- IR alone does not beat fixed chunking here (−0.011), unlike ManiSkill (+0.033 to +0.038). It cannot see pushes
  that happen after the plan is made, which is the model-error term of the bound.
- Adding PE to IR helps (P2), but the bound gate does not beat PE alone (P3) or fixed chunking (P4).
- Offline, the bound gate ranks the calibration plans' regret best on 3 of 5 tasks (AUROC 0.85-0.90 vs IR 0.77-0.83,
  PE 0.68-0.76), but this did not carry over to closed-loop control. The weights are fitted to rank regret over all
  calibration steps, while at run time the decisive events are pushes, which PE alone catches with its whole budget.
- So the domain split is clear: without disturbances (ManiSkill), the decision term dominates and IR is the best
  gate; under exogenous pushes, the model-error term dominates and PE is the best gate. The theory predicts which
  term matters; a fitted sum of the two does not beat the dominant one.
