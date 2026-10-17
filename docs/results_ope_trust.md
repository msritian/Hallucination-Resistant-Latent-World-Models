# OPE trust: does the score predict the estimation error?

| Robot | level | sarsa_mean | A_mean | Aa_mean | D_mean | B_mean | LBA | D | B |
| :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ope_stack_s10 | policy (10) | +0.33 | +0.21 | -0.21 | +0.37 | +0.30 | +0.32 | +0.03 | +0.24 |
| ope_stack_s10 | episode (300) | -0.03 | +0.09 | -0.26 | +0.16 | -0.02 | +0.09 | +0.20 | +0.23 |
| ope_stack_s10 | per-policy error | pi 0.41, pi_bias-0.3 0.07, pi_bias0.15 0.00, pi_bias0.3 0.85, pi_n0.15 0.01, pi_n0.3 0.04, pi_n0.6 0.05, pi_n1.0 0.07, pi_n2.0 0.30, random 0.01 | | | | | | | |
| ope_stack_s20 | policy (10) | -0.03 | +0.20 | +0.03 | +0.16 | +0.18 | +0.04 | +0.08 | +0.03 |
| ope_stack_s20 | episode (300) | +0.01 | +0.22 | -0.34 | +0.22 | +0.15 | +0.12 | +0.20 | +0.16 |
| ope_stack_s20 | per-policy error | pi 0.15, pi_bias-0.3 0.14, pi_bias0.15 0.36, pi_bias0.3 0.59, pi_n0.15 0.24, pi_n0.3 0.22, pi_n0.6 0.40, pi_n1.0 0.26, pi_n2.0 0.15, random 0.05 | | | | | | | |
| ope_stack_s30 | policy (10) | +0.02 | +0.25 | -0.55 | +0.39 | +0.15 | +0.05 | +0.36 | +0.22 |
| ope_stack_s30 | episode (300) | +0.01 | +0.14 | -0.20 | +0.14 | +0.10 | +0.09 | +0.19 | +0.22 |
| ope_stack_s30 | per-policy error | pi 2.01, pi_bias-0.3 0.95, pi_bias0.15 1.32, pi_bias0.3 0.09, pi_n0.15 1.40, pi_n0.3 1.14, pi_n0.6 1.49, pi_n1.0 0.93, pi_n2.0 0.13, random 0.01 | | | | | | | |
| ope_ycb_s10 | policy (10) | +0.64 | +0.77 | -0.32 | +0.58 | +0.75 | +0.64 | +0.78 | +0.68 |
| ope_ycb_s10 | episode (300) | +0.28 | +0.31 | +0.15 | +0.22 | +0.22 | +0.42 | +0.29 | +0.33 |
| ope_ycb_s10 | per-policy error | pi 0.93, pi_bias-0.3 0.71, pi_bias0.15 1.36, pi_bias0.3 0.89, pi_n0.15 1.71, pi_n0.3 1.21, pi_n0.6 1.08, pi_n1.0 0.39, pi_n2.0 0.38, random 0.06 | | | | | | | |
| ope_ycb_s20 | policy (10) | +0.58 | +0.71 | +0.45 | +0.55 | +0.48 | +0.54 | +0.31 | +0.61 |
| ope_ycb_s20 | episode (300) | +0.28 | +0.28 | +0.21 | +0.42 | +0.24 | +0.37 | +0.40 | +0.37 |
| ope_ycb_s20 | per-policy error | pi 0.75, pi_bias-0.3 1.22, pi_bias0.15 0.70, pi_bias0.3 0.93, pi_n0.15 2.21, pi_n0.3 0.51, pi_n0.6 0.98, pi_n1.0 0.40, pi_n2.0 0.21, random 0.05 | | | | | | | |
| ope_ycb_s30 | policy (10) | +0.55 | +0.41 | +0.72 | +0.89 | +0.52 | +0.83 | +0.77 | +0.72 |
| ope_ycb_s30 | episode (300) | +0.39 | +0.38 | +0.26 | +0.26 | +0.32 | +0.44 | +0.40 | +0.47 |
| ope_ycb_s30 | per-policy error | pi 0.78, pi_bias-0.3 1.43, pi_bias0.15 0.60, pi_bias0.3 0.03, pi_n0.15 1.78, pi_n0.3 0.70, pi_n0.6 0.53, pi_n1.0 0.47, pi_n2.0 0.29, random 0.04 | | | | | | | |
| **mean (policy level)** | | **+0.35** | **+0.42** | **+0.02** | **+0.49** | **+0.39** | **+0.40** | **+0.39** | **+0.42** |
