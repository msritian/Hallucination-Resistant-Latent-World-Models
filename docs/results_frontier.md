# Frontier study: results

Pre-registration: docs/preregistration_frontier.md (commit eb66d87). 12 robots (the confirmation robots), 19 new arms
per robot added to the confirmation arms, 100 evaluation episodes per arm (seeds 6000-6099). HTCondor cluster 8342125.
Every claim is reported.

## Summary

| Claim | Result | 95% CI | Verdict |
| :-- | :-: | :-: | :-: |
| G1 IR@2 − random@2 | +0.047 | [+0.022, +0.074] | pass |
| G1 IR@6 − random@6 | +0.062 | [+0.033, +0.092] | pass |
| G2 IR@2 − k2 | +0.032 | [+0.006, +0.058] | pass |
| G2 IR@6 − k6 | +0.056 | [+0.027, +0.087] | pass |
| G3a IR@4 − FI@4 (flow-instability analogue, Hansen & Wang) | +0.044 | [+0.016, +0.072] | pass |
| G3b IR@4 − RS@4 (rank instability, 2611.11780) | +0.126 | [+0.097, +0.153] | pass |
| G4 IR1@4 (single head) − random@4 | +0.080 | [+0.050, +0.111] | pass |

**Success against planning compute (mean over 12 robots):**

| Gate | 1 action/plan | 2 | 4 | 6 |
| :-- | :-: | :-: | :-: | :-: |
| fixed k | 0.743 | 0.695 | 0.635 | 0.542 |
| random | 0.743 | 0.679 | 0.577 | 0.535 |
| return-error oracle | 0.743 | 0.693 | 0.631 | 0.568 |
| act oracle | 0.743 | 0.720 | 0.679 | 0.602 |
| reg oracle | 0.743 | 0.728 | 0.722 | 0.647 |
| **IR (label-free, before acting)** | 0.743 | **0.727** | **0.667** | **0.598** |
| LBA | 0.743 | 0.701 | 0.612 | 0.546 |
| PE (sees the real state) | 0.743 | 0.747 | 0.654 | 0.567 |

**Reading.**
- IR beats random and fixed chunking at every budget, and matches the act oracle at every budget
  (IR − act: +0.007 at 2, −0.012 at 4, −0.004 at 6; all CIs include 0).
- At half the planning compute, IR is statistically indistinguishable from replanning after every action
  (IR@2 − k1 = −0.017 [−0.042, +0.008]).
- The return-error oracle gives no gain over fixed chunking at any budget (−0.002 at 2, +0.026 at 6, CIs include 0).
  Perfect detection of the usual hallucination label buys almost nothing; detecting decision regret does.
- The ensemble is not needed: IR − IR1 = +0.011 [−0.018, +0.038].
- Rank instability (RS) is worse than random (−0.035): ranking flips among near-tied candidates are frequent and
  harmless. FI helps (+0.047 over random) but less than IR.
- PE (which sees the real state while acting) is best at budget 2 (0.747) but falls behind IR at 4 and 6, where blind
  steps are long and the gate must decide before acting.
- Accuracy and control disagree again: on return-error labels, LBA AUROC 0.865 vs IR 0.716, yet IR is the better gate
  at every budget (IR − LBA: +0.026 at 2, +0.055 at 4, +0.052 at 6).

Caveat: achieved actions per plan for fixed k6 is 5.56 (episodes end mid-plan), against 5.76 for IR@6; IR therefore
used slightly less planning compute than k6, which makes G2 at budget 6 conservative.

## Full report (`python -m src.tools.commit_report`, sections 6-7 are this study)

12 robots. Success = task success over the same evaluation episodes for every arm; actions/plan = executed actions per planning call (compute = 1 / actions/plan).

## 1. Success per arm

| Arm | fr_mat_grounded_push_s60 | fr_mat_grounded_push_s61 | fr_mat_grounded_push_s62 | fr_mat_grounded_stack_s60 | fr_mat_grounded_stack_s61 | fr_mat_grounded_stack_s62 | fr_mat_grounded_ycb_s60 | fr_mat_grounded_ycb_s61 | fr_mat_grounded_ycb_s62 | fr_seed_grounded_push_s50 | fr_seed_grounded_stack_s50 | fr_seed_grounded_ycb_s50 | Mean | Actions/plan |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| k1 | 0.96 | 0.99 | 0.98 | 0.84 | 0.74 | 0.90 | 0.29 | 0.38 | 0.40 | 0.98 | 0.88 | 0.58 | **0.743** | 1.00 |
| k2 | 0.96 | 0.96 | 0.99 | 0.70 | 0.67 | 0.83 | 0.25 | 0.35 | 0.39 | 0.98 | 0.85 | 0.41 | **0.695** | 2.00 |
| k4 | 0.90 | 0.91 | 0.99 | 0.62 | 0.57 | 0.73 | 0.19 | 0.32 | 0.31 | 0.94 | 0.71 | 0.43 | **0.635** | 3.85 |
| k6 | 0.94 | 0.93 | 0.96 | 0.42 | 0.33 | 0.50 | 0.19 | 0.25 | 0.27 | 0.86 | 0.50 | 0.35 | **0.542** | 5.56 |
| IR@2 | 0.96 | 0.95 | 0.98 | 0.81 | 0.74 | 0.83 | 0.31 | 0.43 | 0.38 | 0.95 | 0.82 | 0.56 | **0.727** | 2.02 |
| LBA@2 | 0.97 | 0.93 | 0.99 | 0.76 | 0.65 | 0.81 | 0.25 | 0.47 | 0.38 | 0.95 | 0.82 | 0.43 | **0.701** | 2.01 |
| PE@2 | 0.97 | 0.98 | 1.00 | 0.76 | 0.79 | 0.91 | 0.36 | 0.42 | 0.38 | 0.97 | 0.86 | 0.56 | **0.747** | 2.00 |
| act@2 | 0.97 | 0.96 | 0.98 | 0.77 | 0.76 | 0.82 | 0.34 | 0.39 | 0.41 | 0.93 | 0.84 | 0.47 | **0.720** | 2.00 |
| oracle@2 | 0.97 | 1.00 | 0.98 | 0.59 | 0.63 | 0.87 | 0.24 | 0.41 | 0.41 | 0.97 | 0.78 | 0.47 | **0.693** | 2.00 |
| random@2 | 0.96 | 0.95 | 1.00 | 0.73 | 0.64 | 0.80 | 0.26 | 0.32 | 0.37 | 0.97 | 0.71 | 0.44 | **0.679** | 1.98 |
| reg@2 | 0.93 | 0.98 | 0.97 | 0.86 | 0.80 | 0.84 | 0.27 | 0.44 | 0.32 | 0.99 | 0.88 | 0.45 | **0.728** | 2.00 |
| Aa@4 | 0.97 | 0.94 | 1.00 | 0.57 | 0.65 | 0.71 | 0.21 | 0.33 | 0.31 | 0.95 | 0.66 | 0.39 | **0.641** | 3.96 |
| D@4 | 0.94 | 0.96 | 0.98 | 0.47 | 0.47 | 0.63 | 0.18 | 0.25 | 0.24 | 0.93 | 0.47 | 0.41 | **0.578** | 3.98 |
| DBA@4 | 0.94 | 0.95 | 0.97 | 0.66 | 0.56 | 0.66 | 0.28 | 0.26 | 0.29 | 0.91 | 0.75 | 0.46 | **0.641** | 3.94 |
| FI@4 | 0.97 | 0.97 | 0.97 | 0.51 | 0.52 | 0.76 | 0.17 | 0.28 | 0.28 | 0.95 | 0.69 | 0.41 | **0.623** | 3.97 |
| IR1@4 | 0.96 | 0.94 | 0.98 | 0.58 | 0.65 | 0.69 | 0.26 | 0.41 | 0.31 | 0.98 | 0.67 | 0.45 | **0.657** | 3.97 |
| IR@4 | 0.95 | 0.96 | 0.98 | 0.66 | 0.57 | 0.74 | 0.23 | 0.38 | 0.35 | 0.96 | 0.74 | 0.49 | **0.667** | 3.96 |
| LBA@4 | 0.96 | 0.95 | 0.98 | 0.44 | 0.49 | 0.77 | 0.24 | 0.22 | 0.30 | 0.89 | 0.70 | 0.41 | **0.612** | 3.94 |
| M@4 | 0.97 | 0.93 | 0.95 | 0.53 | 0.51 | 0.70 | 0.20 | 0.35 | 0.33 | 0.95 | 0.67 | 0.39 | **0.623** | 3.98 |
| PD@4 | 0.95 | 0.96 | 0.93 | 0.53 | 0.39 | 0.71 | 0.24 | 0.37 | 0.24 | 0.90 | 0.64 | 0.46 | **0.610** | 3.94 |
| PE@4 | 0.92 | 0.97 | 0.98 | 0.64 | 0.65 | 0.76 | 0.24 | 0.25 | 0.34 | 0.94 | 0.71 | 0.45 | **0.654** | 3.97 |
| RS@4 | 0.95 | 0.94 | 0.96 | 0.40 | 0.35 | 0.48 | 0.19 | 0.21 | 0.28 | 0.92 | 0.50 | 0.32 | **0.542** | 3.98 |
| act@4 | 0.93 | 0.98 | 0.98 | 0.61 | 0.72 | 0.75 | 0.24 | 0.40 | 0.35 | 0.92 | 0.79 | 0.48 | **0.679** | 3.96 |
| oracle@4 | 0.94 | 1.00 | 0.97 | 0.64 | 0.57 | 0.69 | 0.20 | 0.30 | 0.25 | 0.90 | 0.70 | 0.41 | **0.631** | 3.94 |
| random@4 | 0.93 | 0.91 | 0.96 | 0.50 | 0.43 | 0.56 | 0.19 | 0.27 | 0.30 | 0.90 | 0.56 | 0.41 | **0.577** | 3.87 |
| reg@4 | 0.95 | 0.99 | 0.96 | 0.80 | 0.74 | 0.90 | 0.23 | 0.40 | 0.40 | 0.98 | 0.82 | 0.49 | **0.722** | 3.98 |
| IR@6 | 0.96 | 0.97 | 0.97 | 0.46 | 0.45 | 0.66 | 0.16 | 0.32 | 0.28 | 0.89 | 0.66 | 0.39 | **0.598** | 5.76 |
| LBA@6 | 0.95 | 0.96 | 0.97 | 0.34 | 0.50 | 0.46 | 0.16 | 0.30 | 0.23 | 0.89 | 0.45 | 0.34 | **0.546** | 5.73 |
| PE@6 | 0.92 | 0.95 | 0.97 | 0.36 | 0.45 | 0.71 | 0.16 | 0.32 | 0.27 | 0.90 | 0.44 | 0.35 | **0.567** | 5.74 |
| act@6 | 0.90 | 0.93 | 0.97 | 0.55 | 0.39 | 0.76 | 0.21 | 0.26 | 0.28 | 0.88 | 0.67 | 0.42 | **0.602** | 5.77 |
| oracle@6 | 0.93 | 0.96 | 0.97 | 0.45 | 0.39 | 0.61 | 0.13 | 0.23 | 0.22 | 0.91 | 0.61 | 0.40 | **0.568** | 5.75 |
| random@6 | 0.97 | 0.92 | 0.98 | 0.31 | 0.34 | 0.52 | 0.20 | 0.25 | 0.26 | 0.94 | 0.42 | 0.31 | **0.535** | 5.66 |
| reg@6 | 0.98 | 0.97 | 0.99 | 0.64 | 0.52 | 0.78 | 0.22 | 0.33 | 0.32 | 0.94 | 0.71 | 0.36 | **0.647** | 5.78 |

## 2a. Accuracy of each gate score on real return errors (calibration plans, steps 1-7, AUROC)

| Robot | LBA | DBA | Aa | D | M | PD | IR | PE |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| fr_mat_grounded_push_s60 | 0.959 | 0.632 | 0.966 | 0.856 | 0.875 | 0.721 | 0.720 | 0.877 |
| fr_mat_grounded_push_s61 | 0.979 | 0.278 | 0.975 | 0.553 | 0.916 | 0.788 | 0.901 | 0.575 |
| fr_mat_grounded_push_s62 | 0.988 | 0.364 | 0.987 | 0.330 | 0.864 | 0.406 | 0.953 | 0.343 |
| fr_mat_grounded_stack_s60 | 0.851 | 0.819 | 0.809 | 0.859 | 0.776 | 0.740 | 0.766 | 0.964 |
| fr_mat_grounded_stack_s61 | 0.867 | 0.872 | 0.864 | 0.810 | 0.759 | 0.669 | 0.764 | 0.920 |
| fr_mat_grounded_stack_s62 | 0.840 | 0.854 | 0.831 | 0.804 | 0.764 | 0.771 | 0.707 | 0.931 |
| fr_mat_grounded_ycb_s60 | 0.785 | 0.811 | 0.865 | 0.681 | 0.709 | 0.747 | 0.790 | 0.864 |
| fr_mat_grounded_ycb_s61 | 0.887 | 0.744 | 0.692 | 0.784 | 0.708 | 0.707 | 0.722 | 0.916 |
| fr_mat_grounded_ycb_s62 | 0.920 | 0.527 | 0.723 | 0.688 | 0.666 | 0.575 | 0.549 | 0.685 |
| fr_seed_grounded_push_s50 | 0.909 | 0.476 | 0.924 | 0.733 | 0.778 | 0.786 | 0.788 | 0.871 |
| fr_seed_grounded_stack_s50 | 0.848 | 0.796 | 0.838 | 0.744 | 0.795 | 0.588 | 0.778 | 0.885 |
| fr_seed_grounded_ycb_s50 | 0.857 | 0.548 | 0.790 | 0.471 | 0.679 | 0.610 | 0.662 | 0.919 |
| **Mean** | **0.891** | **0.643** | **0.855** | **0.693** | **0.774** | **0.676** | **0.758** | **0.813** |

## 2b. Accuracy of each gate score on decision-relevant errors (act oracle) (calibration plans, steps 1-7, AUROC)

| Robot | LBA | DBA | Aa | D | M | PD | IR | PE |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| fr_mat_grounded_push_s60 | 0.547 | 0.582 | 0.533 | 0.532 | 0.480 | 0.532 | 0.474 | 0.580 |
| fr_mat_grounded_push_s61 | 0.436 | 0.535 | 0.424 | 0.506 | 0.404 | 0.495 | 0.467 | 0.543 |
| fr_mat_grounded_push_s62 | 0.499 | 0.561 | 0.486 | 0.489 | 0.471 | 0.540 | 0.514 | 0.495 |
| fr_mat_grounded_stack_s60 | 0.832 | 0.885 | 0.826 | 0.829 | 0.782 | 0.716 | 0.894 | 0.873 |
| fr_mat_grounded_stack_s61 | 0.905 | 0.954 | 0.913 | 0.927 | 0.920 | 0.805 | 0.905 | 0.951 |
| fr_mat_grounded_stack_s62 | 0.823 | 0.789 | 0.810 | 0.747 | 0.704 | 0.742 | 0.774 | 0.766 |
| fr_mat_grounded_ycb_s60 | 0.491 | 0.703 | 0.625 | 0.655 | 0.666 | 0.668 | 0.737 | 0.742 |
| fr_mat_grounded_ycb_s61 | 0.604 | 0.733 | 0.590 | 0.703 | 0.664 | 0.640 | 0.705 | 0.766 |
| fr_mat_grounded_ycb_s62 | 0.529 | 0.596 | 0.541 | 0.643 | 0.560 | 0.653 | 0.565 | 0.741 |
| fr_seed_grounded_push_s50 | 0.554 | 0.582 | 0.558 | 0.585 | 0.514 | 0.581 | 0.537 | 0.624 |
| fr_seed_grounded_stack_s50 | 0.797 | 0.881 | 0.780 | 0.827 | 0.797 | 0.787 | 0.841 | 0.862 |
| fr_seed_grounded_ycb_s50 | 0.623 | 0.653 | 0.699 | 0.593 | 0.706 | 0.618 | 0.745 | 0.803 |
| **Mean** | **0.637** | **0.705** | **0.649** | **0.670** | **0.639** | **0.648** | **0.680** | **0.729** |

(PE uses the real next state, so it is only available while acting, not before.)

## 3. Pre-registered claims (confirmation)

- **K1** act@4 − random@4 = +0.102 [+0.073, +0.133] (12 robots) → passed
- **K2** act@4 − oracle@4 = +0.048 [+0.019, +0.078] (12 robots) → passed
- **K3** Aa@4 − random@4 = +0.064 [+0.037, +0.093] (12 robots); PE@4 − random@4 = +0.077 [+0.049, +0.107] (12 robots) → passed
- **K4** DBA@4 − random@4 = +0.064 [+0.034, +0.093] (12 robots); DBA@4 − LBA@4 = +0.028 [-0.000, +0.057] (12 robots) → passed

## 4. Reported, not claimed

- reg@4 − random@4: +0.145 [+0.116, +0.174] (12 robots)
- act@4 − reg@4: -0.043 [-0.071, -0.014] (12 robots)
- oracle@4 − random@4: +0.054 [+0.025, +0.083] (12 robots)
- LBA@4 − random@4: +0.036 [+0.005, +0.066] (12 robots)
- D@4 − random@4: +0.001 [-0.029, +0.032] (12 robots)
- M@4 − random@4: +0.047 [+0.016, +0.078] (12 robots)
- DBA@4 − Aa@4: +0.000 [-0.028, +0.029] (12 robots)
- DBA@4 − PE@4: -0.013 [-0.043, +0.015] (12 robots)
- k1 − k4: +0.108 [+0.082, +0.133] (12 robots)
- act@4 − k1: -0.064 [-0.091, -0.037] (12 robots)

## 5. Pre-registered claims (decision-flip signals)

- **F1** IR@4 − random@4 = +0.091 [+0.062, +0.121] (12 robots) → passed
- **F2** PD@4 − D@4 = +0.032 [+0.002, +0.062] (12 robots) → passed
- **F3** IR@4 − LBA@4 = +0.055 [+0.027, +0.083] (12 robots) → passed

Reported, not claimed:

- PD@4 − random@4: +0.033 [+0.002, +0.065] (12 robots)
- IR@4 − DBA@4: +0.027 [-0.002, +0.055] (12 robots)
- PD@4 − DBA@4: -0.031 [-0.061, -0.002] (12 robots)
- IR@4 − Aa@4: +0.027 [-0.001, +0.053] (12 robots)
- IR@4 − PE@4: +0.013 [-0.016, +0.043] (12 robots)
- IR@4 − act@4: -0.012 [-0.040, +0.018] (12 robots)
- PD@4 − act@4: -0.069 [-0.099, -0.040] (12 robots)

## 6. Success against planning compute (frontier study)

| Gate | budget 1 | budget 2 | budget 4 | budget 6 |
| :-- | :-: | :-: | :-: | :-: |
| fixed k | 0.743 | 0.695 | 0.635 | 0.542 |
| random | 0.743 | 0.679 | 0.577 | 0.535 |
| oracle | 0.743 | 0.693 | 0.631 | 0.568 |
| act | 0.743 | 0.720 | 0.679 | 0.602 |
| reg | 0.743 | 0.728 | 0.722 | 0.647 |
| IR | 0.743 | 0.727 | 0.667 | 0.598 |
| LBA | 0.743 | 0.701 | 0.612 | 0.546 |
| PE | 0.743 | 0.747 | 0.654 | 0.567 |

(budget 1 = replanning after every action, k1, for every row; budget = mean executed actions per plan.)

## 7. Pre-registered claims (frontier)

- **G1** IR@2 − random@2 = +0.047 [+0.022, +0.074] (12 robots); IR@6 − random@6 = +0.062 [+0.033, +0.092] (12 robots) → passed
- **G2** IR@2 − k2 = +0.032 [+0.006, +0.058] (12 robots); IR@6 − k6 = +0.056 [+0.027, +0.087] (12 robots) → passed
- **G3a** IR@4 − FI@4 = +0.044 [+0.016, +0.072] (12 robots) → passed
- **G3b** IR@4 − RS@4 = +0.126 [+0.097, +0.153] (12 robots) → passed
- **G4** IR1@4 − random@4 = +0.080 [+0.050, +0.111] (12 robots) → passed

Reported, not claimed:

- act@2 − random@2: +0.041 [+0.014, +0.068] (12 robots)
- reg@2 − random@2: +0.048 [+0.022, +0.073] (12 robots)
- oracle@2 − random@2: +0.014 [-0.013, +0.042] (12 robots)
- LBA@2 − random@2: +0.022 [-0.005, +0.049] (12 robots)
- PE@2 − random@2: +0.068 [+0.041, +0.093] (12 robots)
- act@6 − random@6: +0.067 [+0.037, +0.095] (12 robots)
- reg@6 − random@6: +0.112 [+0.083, +0.140] (12 robots)
- oracle@6 − random@6: +0.032 [+0.003, +0.062] (12 robots)
- LBA@6 − random@6: +0.011 [-0.018, +0.038] (12 robots)
- PE@6 − random@6: +0.032 [+0.003, +0.060] (12 robots)
- act@2 − k2: +0.025 [-0.002, +0.052] (12 robots)
- reg@2 − k2: +0.032 [+0.008, +0.057] (12 robots)
- oracle@2 − k2: -0.002 [-0.030, +0.027] (12 robots)
- LBA@2 − k2: +0.006 [-0.021, +0.032] (12 robots)
- PE@2 − k2: +0.052 [+0.026, +0.077] (12 robots)
- random@2 − k2: -0.016 [-0.043, +0.010] (12 robots)
- act@6 − k6: +0.060 [+0.029, +0.091] (12 robots)
- reg@6 − k6: +0.105 [+0.074, +0.135] (12 robots)
- oracle@6 − k6: +0.026 [-0.004, +0.056] (12 robots)
- LBA@6 − k6: +0.004 [-0.026, +0.035] (12 robots)
- PE@6 − k6: +0.025 [-0.004, +0.055] (12 robots)
- random@6 − k6: -0.007 [-0.036, +0.022] (12 robots)
- IR@2 − act@2: +0.007 [-0.021, +0.033] (12 robots)
- IR@2 − LBA@2: +0.026 [-0.001, +0.052] (12 robots)
- IR@6 − act@6: -0.004 [-0.036, +0.027] (12 robots)
- IR@6 − LBA@6: +0.052 [+0.023, +0.080] (12 robots)
- IR@4 − IR1@4: +0.011 [-0.018, +0.038] (12 robots)
- FI@4 − random@4: +0.047 [+0.018, +0.075] (12 robots)
- RS@4 − random@4: -0.035 [-0.063, -0.006] (12 robots)
- IR@4 − k4: +0.032 [+0.003, +0.061] (12 robots)
- k1 − k2: +0.048 [+0.023, +0.073] (12 robots)
- k1 − k6: +0.202 [+0.173, +0.230] (12 robots)
- IR@2 − k1: -0.017 [-0.042, +0.008] (12 robots)

Accuracy of the new scores on real return errors (frontier calibration plans, steps 1-7, AUROC):

| Robot | IR | IR1 | FI | RS | LBA |
| :-- | :-: | :-: | :-: | :-: | :-: |
| fr_mat_grounded_push_s60 | 0.712 | 0.630 | 0.735 | 0.506 | 0.959 |
| fr_mat_grounded_push_s61 | 0.868 | 0.677 | 0.325 | 0.569 | 0.914 |
| fr_mat_grounded_push_s62 | 0.756 | 0.507 | 0.828 | 0.592 | 0.962 |
| fr_mat_grounded_stack_s60 | 0.746 | 0.637 | 0.772 | 0.469 | 0.787 |
| fr_mat_grounded_stack_s61 | 0.765 | 0.718 | 0.717 | 0.444 | 0.882 |
| fr_mat_grounded_stack_s62 | 0.725 | 0.686 | 0.834 | 0.428 | 0.894 |
| fr_mat_grounded_ycb_s60 | 0.686 | 0.615 | 0.523 | 0.540 | 0.749 |
| fr_mat_grounded_ycb_s61 | 0.560 | 0.516 | 0.597 | 0.503 | 0.781 |
| fr_mat_grounded_ycb_s62 | 0.452 | 0.413 | 0.436 | 0.563 | 0.891 |
| fr_seed_grounded_push_s50 | 0.921 | 0.872 | 0.356 | 0.653 | 0.967 |
| fr_seed_grounded_stack_s50 | 0.805 | 0.749 | 0.765 | 0.398 | 0.846 |
| fr_seed_grounded_ycb_s50 | 0.594 | 0.512 | 0.613 | 0.458 | 0.753 |
| **Mean** | **0.716** | **0.628** | **0.625** | **0.510** | **0.865** |

Actions per plan: k2 2.00, random@2 1.98, IR@2 2.02, act@2 2.00, k6 5.56, random@6 5.66, IR@6 5.76, act@6 5.77, IR1@4 3.97, FI@4 3.97, RS@4 3.98
