# Commitment study: results (pre-registered in docs/preregistration_commitment.md, amendment 1)

Tool: `python -m src.tools.commit_report com_*.tar.gz`. 10 robots, 19 arms, 100 episodes per arm, same seeds for all arms.

## Summary (read this first)

- **Committing to the plan hurts a lot**, as hypothesised: success falls from 0.620 (replan every step) to 0.492
  (4 actions per plan) and 0.387 (8 actions per plan).
- **But a perfect detector of our hallucination label barely helps choosing when to commit**: oracle@4 − random@4 =
  +0.029 [−0.004, +0.063]. C0 fails, so by the pre-registration C1–C3 are reported and not interpreted. C3 is
  formally "passed" (mean +0.002 > 0), but that mean is zero within noise and we do not claim it.
- The LBA captures essentially all of that small oracle gain (+0.028 vs +0.029 at budget 4), but there is little to
  capture.
- **Exploratory (not pre-registered, needs confirmation):** two gates beat random at budget 4 by about twice the
  oracle's gain: the label-free anchored residual Aa (+0.054 [+0.021, +0.086]) and the real-observation check PE
  (+0.056 [+0.022, +0.089]). So the plans worth cutting short are not the same plans our return-error label marks as
  hallucinated. The label, not the detector, is the limit for control.
- LBA+PE@4 is +0.082 over PE@4, but it committed to only 2.39 actions per plan (not compute-matched), so this
  comparison is not fair and we do not use it.

## Full output

10 robots. Success = task success over the same evaluation episodes for every arm; actions/plan = executed actions per planning call (compute = 1 / actions/plan).

## 1. Success per arm

| Arm | bk_grounded_stackcube_s10 | p1_grounded_push_s10 | p4_grounded_ycb_s10 | seed_grounded_push_s40 | seed_grounded_stack_s20 | seed_grounded_stack_s30 | seed_grounded_stack_s40 | seed_grounded_ycb_s20 | seed_grounded_ycb_s30 | seed_grounded_ycb_s40 | Mean | Actions/plan |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| k1 | 0.69 | 0.91 | 0.41 | 1.00 | 0.83 | 0.67 | 0.71 | 0.43 | 0.35 | 0.20 | **0.620** | 1.00 |
| k2 | 0.66 | 0.96 | 0.37 | 0.95 | 0.83 | 0.62 | 0.66 | 0.32 | 0.28 | 0.13 | **0.578** | 2.00 |
| k4 | 0.63 | 0.94 | 0.29 | 0.94 | 0.65 | 0.36 | 0.49 | 0.30 | 0.24 | 0.08 | **0.492** | 3.85 |
| k8 | 0.31 | 0.88 | 0.29 | 0.94 | 0.47 | 0.19 | 0.30 | 0.25 | 0.14 | 0.10 | **0.387** | 7.14 |
| Aa@2 | 0.69 | 0.97 | 0.42 | 1.00 | 0.82 | 0.65 | 0.72 | 0.33 | 0.25 | 0.14 | **0.599** | 2.01 |
| D@2 | 0.68 | 0.95 | 0.44 | 0.97 | 0.91 | 0.55 | 0.67 | 0.36 | 0.21 | 0.14 | **0.588** | 2.02 |
| LBA@2 | 0.66 | 0.98 | 0.40 | 0.95 | 0.93 | 0.65 | 0.74 | 0.38 | 0.19 | 0.15 | **0.603** | 2.01 |
| M@2 | 0.77 | 0.97 | 0.38 | 0.95 | 0.80 | 0.49 | 0.79 | 0.34 | 0.32 | 0.14 | **0.595** | 2.03 |
| PE@2 | 0.72 | 0.94 | 0.40 | 0.99 | 0.85 | 0.72 | 0.67 | 0.40 | 0.23 | 0.12 | **0.604** | 2.00 |
| oracle@2 | 0.73 | 0.96 | 0.43 | 0.98 | 0.82 | 0.55 | 0.66 | 0.30 | 0.20 | 0.13 | **0.576** | 1.99 |
| random@2 | 0.65 | 0.91 | 0.39 | 0.98 | 0.78 | 0.62 | 0.66 | 0.37 | 0.24 | 0.17 | **0.577** | 1.90 |
| Aa@4 | 0.61 | 0.96 | 0.40 | 1.00 | 0.62 | 0.51 | 0.51 | 0.22 | 0.24 | 0.13 | **0.520** | 3.97 |
| D@4 | 0.51 | 0.96 | 0.33 | 0.97 | 0.68 | 0.39 | 0.55 | 0.31 | 0.11 | 0.10 | **0.491** | 3.96 |
| LBA+PE@4 | 0.68 | 0.97 | 0.43 | 0.95 | 0.87 | 0.59 | 0.74 | 0.45 | 0.26 | 0.10 | **0.604** | 2.39 |
| LBA@4 | 0.59 | 0.90 | 0.28 | 0.96 | 0.73 | 0.32 | 0.50 | 0.34 | 0.16 | 0.16 | **0.494** | 3.94 |
| M@4 | 0.58 | 0.96 | 0.29 | 0.94 | 0.62 | 0.50 | 0.52 | 0.29 | 0.19 | 0.11 | **0.500** | 3.97 |
| PE@4 | 0.68 | 0.97 | 0.34 | 0.92 | 0.64 | 0.53 | 0.56 | 0.28 | 0.18 | 0.12 | **0.522** | 3.96 |
| oracle@4 | 0.63 | 0.94 | 0.34 | 0.93 | 0.69 | 0.47 | 0.46 | 0.24 | 0.17 | 0.08 | **0.495** | 3.94 |
| random@4 | 0.35 | 0.97 | 0.39 | 0.87 | 0.58 | 0.39 | 0.46 | 0.27 | 0.22 | 0.16 | **0.466** | 3.80 |

## 2. Accuracy of each gate score on real plan errors (calibration plans, steps 1-7, AUROC)

| Robot | LBA | Aa | D | M | PE |
| :-- | :-: | :-: | :-: | :-: | :-: |
| bk_grounded_stackcube_s10 | 0.910 | 0.885 | 0.778 | 0.738 | 0.948 |
| p1_grounded_push_s10 | 0.844 | 0.832 | 0.639 | 0.836 | 0.728 |
| p4_grounded_ycb_s10 | 0.854 | 0.582 | 0.555 | 0.453 | 0.871 |
| seed_grounded_push_s40 | 0.969 | 0.972 | 0.806 | 0.834 | 0.912 |
| seed_grounded_stack_s20 | 0.869 | 0.818 | 0.867 | 0.845 | 0.889 |
| seed_grounded_stack_s30 | 0.734 | 0.685 | 0.781 | 0.754 | 0.846 |
| seed_grounded_stack_s40 | 0.689 | 0.802 | 0.744 | 0.749 | 0.842 |
| seed_grounded_ycb_s20 | 0.670 | 0.840 | 0.553 | 0.608 | 0.865 |
| seed_grounded_ycb_s30 | 0.889 | 0.651 | 0.623 | 0.616 | 0.869 |
| seed_grounded_ycb_s40 | 0.749 | 0.546 | 0.526 | 0.511 | 0.641 |
| **Mean** | **0.818** | **0.761** | **0.687** | **0.694** | **0.841** |

(PE uses the real next state, so it is only available while acting, not before.)

## 3. Pre-registered claims

- **C0** committing hurts: k1 − k4 = +0.128 [+0.096, +0.160] (10 robots); perfect gating helps: oracle@4 − random@4 = +0.029 [-0.004, +0.063] (10 robots) → failed
- **C1** LBA@4 − random@4 = +0.028 [-0.006, +0.062] (10 robots) → failed
- **C2** LBA@4 − D@4 = +0.003 [-0.030, +0.036] (10 robots); LBA@4 − M@4 = -0.006 [-0.039, +0.027] (10 robots) → failed
- **C3** [LBA − random]@4 − [LBA − random]@2 = +0.002 [-0.043, +0.046] (10 robots) → passed

  C0 failed: the setting gives detection little to buy; C1-C3 are reported, not interpreted.

## 4. Reported, not claimed

- LBA@2 − random@2: +0.026 [-0.007, +0.058] (10 robots)
- Aa@4 − random@4: +0.054 [+0.021, +0.086] (10 robots)
- Aa@2 − random@2: +0.022 [-0.008, +0.051] (10 robots)
- LBA@4 − Aa@4: -0.026 [-0.058, +0.006] (10 robots)
- PE@4 − random@4: +0.056 [+0.022, +0.089] (10 robots)
- LBA@4 − PE@4: -0.028 [-0.061, +0.005] (10 robots)
- LBA+PE@4 − PE@4: +0.082 [+0.050, +0.115] (10 robots)
- LBA@4 − k4: +0.002 [-0.031, +0.035] (10 robots)
- oracle@4 − k4: +0.003 [-0.031, +0.037] (10 robots)
- oracle@2 − random@2: -0.001 [-0.033, +0.031] (10 robots)
- k1 − k2: +0.042 [+0.009, +0.073] (10 robots)
- k1 − k8: +0.233 [+0.198, +0.266] (10 robots)
- LBA@4 − k2: -0.084 [-0.116, -0.053] (10 robots)
- LBA@4 − k1: -0.126 [-0.159, -0.092] (10 robots)
- Share of the oracle's gain over random captured by the LBA at budget 2: -26.00
- Share of the oracle's gain over random captured by the LBA at budget 4: 0.97
