# Learned Bellman audit: ablations, data needs, transfer and statistics

22 robots (11 with our critic). Simulator labels; AUROC (0.5 guessing, 1.0 perfect). All audits use the same gradient-boosted tree settings as the main results.

## 1. Which signals matter (average over all robots)

| Audit reads… | All | Gradual | Sudden |
| :-- | :-: | :-: | :-: |
| Full audit (ours) | 0.819 | 0.823 | 0.844 |
| One-step δ only | 0.721 | 0.730 | 0.737 |
| Anchored δ only | 0.724 | 0.728 | 0.739 |
| One-step + anchored δ | 0.741 | 0.746 | 0.755 |
| All residuals, no head disagreement | 0.821 | 0.825 | 0.843 |
| All residuals, no multi-step δ | 0.813 | 0.819 | 0.838 |
| All residuals, no anchored δ | 0.818 | 0.822 | 0.843 |
| No Bellman: value, reward, step only | 0.769 | 0.772 | 0.784 |
| No Bellman: critic spread + value, reward, step | 0.794 | 0.802 | 0.809 |

## 2. How many episodes the audit needs (average over all robots, all hallucinations)

| Calibration episodes | AUROC |
| :-: | :-: |
| 5 | 0.758 |
| 10 | 0.781 |
| 15 | 0.805 |
| 25 | 0.819 |

## 3. Transfer: audit fitted on one robot, used on another (all hallucinations)

| Fitted on | Raw features | Standardised with the target's unlabeled data |
| :-- | :-: | :-: |
| the same robot | 0.819 (22 pairs) | 0.819 |
| another robot, same task | 0.804 (52 pairs) | 0.784 |
| a robot of another task | 0.709 (168 pairs) | 0.612 |

## 4a. Ours minus MOBILE-style (as published), per robot (paired bootstrap over episodes, all hallucinations)

| Robot | Ours | MOBILE-style | Difference | 95% interval | Significant? |
| :-- | :-: | :-: | :-: | :-: | :-: |
| bk_grounded_stackcube_s10 | 0.797 | 0.736 | +0.061 | [+0.016, +0.102] | yes |
| p1_grounded_push_s10 | 0.915 | 0.851 | +0.064 | [-0.004, +0.144] | no |
| p4_grounded_peg_s10 | 0.720 | 0.502 | +0.217 | [+0.112, +0.314] | yes |
| p4_grounded_ycb_s10 | 0.775 | 0.663 | +0.111 | [+0.026, +0.194] | yes |
| seed_grounded_push_s40 | 0.962 | 0.777 | +0.185 | [+0.069, +0.291] | yes |
| seed_grounded_stack_s20 | 0.704 | 0.615 | +0.089 | [-0.049, +0.240] | no |
| seed_grounded_stack_s30 | 0.898 | 0.827 | +0.071 | [+0.027, +0.126] | yes |
| seed_grounded_stack_s40 | 0.754 | 0.797 | -0.043 | [-0.168, +0.077] | no |
| seed_grounded_ycb_s20 | 0.732 | 0.571 | +0.161 | [+0.024, +0.315] | yes |
| seed_grounded_ycb_s30 | 0.884 | 0.473 | +0.411 | [+0.272, +0.564] | yes |
| seed_grounded_ycb_s40 | 0.833 | 0.718 | +0.115 | [+0.031, +0.186] | yes |

Ours significantly better on 8 of 11 robots, significantly worse on 0; mean difference +0.131.

## 4b. Ours minus Dynamics ensemble (as published), per robot (paired bootstrap over episodes, all hallucinations)

| Robot | Ours | Dynamics ensemble | Difference | 95% interval | Significant? |
| :-- | :-: | :-: | :-: | :-: | :-: |
| bk_grounded_stackcube_s10 | 0.797 | 0.764 | +0.033 | [-0.044, +0.103] | no |
| p1_grounded_push_s10 | 0.915 | 0.486 | +0.428 | [+0.283, +0.575] | yes |
| p4_grounded_peg_s10 | 0.720 | 0.510 | +0.210 | [+0.068, +0.380] | yes |
| p4_grounded_ycb_s10 | 0.775 | 0.712 | +0.063 | [+0.015, +0.110] | yes |
| seed_grounded_push_s40 | 0.962 | 0.614 | +0.348 | [+0.205, +0.479] | yes |
| seed_grounded_stack_s20 | 0.704 | 0.628 | +0.076 | [-0.103, +0.295] | no |
| seed_grounded_stack_s30 | 0.898 | 0.812 | +0.086 | [+0.028, +0.143] | yes |
| seed_grounded_stack_s40 | 0.754 | 0.812 | -0.058 | [-0.168, +0.049] | no |
| seed_grounded_ycb_s20 | 0.732 | 0.791 | -0.059 | [-0.216, +0.077] | no |
| seed_grounded_ycb_s30 | 0.884 | 0.461 | +0.423 | [+0.294, +0.558] | yes |
| seed_grounded_ycb_s40 | 0.833 | 0.809 | +0.024 | [-0.050, +0.096] | no |

Ours significantly better on 6 of 11 robots, significantly worse on 0; mean difference +0.143.

## 5. Before vs after the task is already solved (robots with our critic, all hallucinations)

A window counts as 'after' if the real episode had already reached success at the window's start.

| Robot | Ours: before | Ours: after | Dynamics ensemble: before | Dynamics ensemble: after |
| :-- | :-: | :-: | :-: | :-: |
| bk_grounded_stackcube_s10 | 0.749 | 0.815 | 0.685 | 0.774 |
| p1_grounded_push_s10 | 0.645 | 0.913 | 0.520 | 0.449 |
| p4_grounded_peg_s10 | 0.696 | – | 0.552 | – |
| p4_grounded_ycb_s10 | 0.780 | 0.802 | 0.718 | 0.562 |
| seed_grounded_push_s40 | 0.727 | 0.979 | 0.587 | 0.591 |
| seed_grounded_stack_s20 | 0.816 | 0.274 | 0.600 | 0.524 |
| seed_grounded_stack_s30 | 0.773 | 1.000 | 0.650 | 0.997 |
| seed_grounded_stack_s40 | 0.800 | 0.656 | 0.779 | 0.805 |
| seed_grounded_ycb_s20 | 0.782 | 0.782 | 0.781 | 0.947 |
| seed_grounded_ycb_s30 | 0.861 | 0.897 | 0.547 | 0.529 |
| seed_grounded_ycb_s40 | 0.837 | 0.781 | 0.816 | 0.692 |

