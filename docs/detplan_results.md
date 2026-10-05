# Detector in the planner: results

11 robots, success rate on the same evaluation seeds.

## Success rate per arm

| Arm | bk_grounded_stackcube_s10 | p1_grounded_push_s10 | p4_grounded_peg_s10 | p4_grounded_ycb_s10 | seed_grounded_push_s40 | seed_grounded_stack_s20 | seed_grounded_stack_s30 | seed_grounded_stack_s40 | seed_grounded_ycb_s20 | seed_grounded_ycb_s30 | seed_grounded_ycb_s40 | Mean |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| L08_H12 | 0.89 | 0.97 | 0.06 | 0.52 | 0.96 | 0.97 | 0.84 | 0.86 | 0.51 | 0.36 | 0.24 | **0.653** |
| L08_H12_drop25 | 0.91 | 1.00 | 0.03 | 0.32 | 0.99 | 0.98 | 0.85 | 0.93 | 0.35 | 0.20 | 0.14 | **0.609** |
| L08_H12_dropRandom25 | 0.88 | 0.98 | 0.05 | 0.46 | 0.96 | 0.96 | 0.88 | 0.92 | 0.45 | 0.37 | 0.21 | **0.647** |
| L08_H24 | 0.85 | 0.97 | 0.05 | 0.51 | 0.99 | 0.93 | 0.89 | 0.85 | 0.49 | 0.34 | 0.22 | **0.645** |
| L08_H3 | 0.92 | 0.98 | 0.07 | 0.53 | 0.98 | 0.95 | 0.92 | 0.90 | 0.41 | 0.34 | 0.23 | **0.657** |
| LBAlam_H12 | 0.89 | 0.99 | 0.03 | 0.57 | 0.98 | 0.88 | 0.84 | 0.90 | 0.44 | 0.38 | 0.21 | **0.646** |
| LBAlam_H12_shuffled | 0.92 | 0.98 | 0.07 | 0.50 | 0.97 | 0.92 | 0.91 | 0.84 | 0.43 | 0.33 | 0.26 | **0.648** |
| LBAlam_H24 | 0.88 | 0.98 | 0.05 | 0.46 | 0.96 | 0.92 | 0.91 | 0.86 | 0.49 | 0.32 | 0.22 | **0.641** |
| LBAlam_pure_H12 | 0.86 | 0.99 | 0.03 | 0.50 | 0.99 | 0.83 | 0.82 | 0.83 | 0.42 | 0.36 | 0.24 | **0.625** |
| standard_H12 | 0.68 | 0.99 | 0.05 | 0.43 | 0.97 | 0.86 | 0.74 | 0.72 | 0.42 | 0.36 | 0.16 | **0.580** |
| standard_H3 | 0.90 | 0.96 | 0.07 | 0.48 | 0.92 | 0.91 | 0.85 | 0.84 | 0.42 | 0.31 | 0.20 | **0.624** |

## Pre-registered claims

- **P1** LBAlam_H12 − L08_H12: -0.006 [-0.030, +0.017] (11 robots) → failed
- **P2** LBAlam_H12 − LBAlam_H12_shuffled: -0.002 [-0.025, +0.022] (11 robots) → failed
- **P3** L08_H12_drop25 − L08_H12: -0.044 [-0.068, -0.020] (11 robots) → failed
- **P4** L08_H12_drop25 − L08_H12_dropRandom25: -0.038 [-0.060, -0.017] (11 robots) → failed

## Exploratory (reported, not claimed)

- LBAlam_pure_H12 − L08_H12: -0.028 [-0.054, -0.003] (11 robots)
- LBAlam_H24 − L08_H24: -0.004 [-0.027, +0.020] (11 robots)
- L08_H24 − L08_H12: -0.008 [-0.033, +0.015] (11 robots)
- LBAlam_H24 − LBAlam_H12: -0.005 [-0.029, +0.017] (11 robots)
- LBAlam_H12 − L08_H3: -0.011 [-0.032, +0.011] (11 robots)
- LBAlam_H12 − standard_H3: +0.023 [-0.001, +0.046] (11 robots)
- LBAlam_H12 − standard_H12: +0.066 [+0.041, +0.094] (11 robots)
- L08_H12 − standard_H3: +0.029 [+0.005, +0.053] (11 robots)
