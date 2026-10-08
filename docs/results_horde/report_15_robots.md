# Horde per-signal audits: results

| Robot | Horde, reward-free | Ensemble D (reward-free) | Reward audit (ours) | Reward audit + Horde | Per-signal AUROC | Localise: Horde | Localise: ensemble | Localise: majority | Localise: chance |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| bk_grounded_stackcube_s10 | 0.762 | 0.764 | 0.797 | 0.822 | 0.723 | 0.074 | 0.074 | 0.926 | 0.021 |
| p1_grounded_push_s10 | 0.704 | 0.486 | 0.915 | 0.905 | 0.638 | 0.180 | 0.116 | 0.240 | 0.029 |
| p4_grounded_peg_s10 | 0.561 | 0.510 | 0.720 | 0.718 | 0.635 | nan | nan | nan | 0.023 |
| p4_grounded_ycb_s10 | 0.703 | 0.712 | 0.775 | 0.806 | 0.649 | 0.542 | 0.688 | 1.000 | 0.022 |
| seed_grounded_peg_s50 | 0.801 | 0.780 | 0.837 | 0.847 | 0.739 | nan | nan | nan | 0.023 |
| seed_grounded_push_s40 | 0.632 | 0.614 | 0.962 | 0.963 | 0.598 | nan | nan | nan | 0.029 |
| seed_grounded_push_s50 | 0.722 | 0.524 | 0.897 | 0.878 | 0.638 | 0.233 | 0.302 | 0.814 | 0.029 |
| seed_grounded_stack_s20 | 0.659 | 0.628 | 0.708 | 0.721 | 0.716 | 0.182 | 0.273 | 0.000 | 0.021 |
| seed_grounded_stack_s30 | 0.790 | 0.812 | 0.898 | 0.911 | 0.687 | 0.179 | 0.333 | 0.057 | 0.021 |
| seed_grounded_stack_s40 | 0.711 | 0.812 | 0.754 | 0.766 | 0.742 | 0.268 | 0.212 | 0.113 | 0.021 |
| seed_grounded_stack_s50 | 0.808 | 0.833 | 0.879 | 0.866 | 0.715 | 0.189 | 0.189 | 1.000 | 0.021 |
| seed_grounded_ycb_s20 | 0.617 | 0.791 | 0.732 | 0.740 | 0.632 | 0.667 | 0.702 | 0.879 | 0.022 |
| seed_grounded_ycb_s30 | 0.735 | 0.461 | 0.884 | 0.904 | 0.630 | 0.633 | 0.838 | 0.948 | 0.022 |
| seed_grounded_ycb_s40 | 0.695 | 0.809 | 0.833 | 0.827 | 0.602 | 0.040 | 0.144 | 0.040 | 0.022 |
| seed_grounded_ycb_s50 | 0.639 | 0.590 | 0.641 | 0.644 | 0.622 | nan | nan | nan | 0.022 |
| **Mean** | **0.703** | **0.675** | **0.816** | **0.821** | **0.664** | **nan** | **nan** | **nan** | **0.023** |

## Signals most often the wrong one (hallucinated windows)

- bk_grounded_stackcube_s10: 36 (25), 40 (1), 39 (1), 25 (0), 24 (0)
- p1_grounded_push_s10: 30 (114), 33 (56), 22 (40), 16 (18), 17 (4)
- p4_grounded_peg_s10: 
- p4_grounded_ycb_s10: 22 (192), 34 (0), 24 (0), 25 (0), 26 (0)
- seed_grounded_peg_s50: 
- seed_grounded_push_s40: 
- seed_grounded_push_s50: 32 (35), 31 (8), 26 (0), 18 (0), 19 (0)
- seed_grounded_stack_s20: 30 (8), 29 (3), 36 (0), 25 (0), 26 (0)
- seed_grounded_stack_s30: 36 (53), 37 (37), 29 (19), 28 (7), 30 (7)
- seed_grounded_stack_s40: 36 (115), 37 (37), 17 (26), 34 (19), 32 (18)
- seed_grounded_stack_s50: 37 (37), 36 (0), 25 (0), 26 (0), 27 (0)
- seed_grounded_ycb_s20: 22 (124), 32 (17), 34 (0), 24 (0), 25 (0)
- seed_grounded_ycb_s30: 22 (217), 28 (9), 37 (2), 13 (1), 33 (0)
- seed_grounded_ycb_s40: 36 (84), 34 (19), 39 (11), 22 (5), 21 (3)
- seed_grounded_ycb_s50: 
