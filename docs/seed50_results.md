# Seed-50 confirmation: results (pre-registered in docs/preregistration_seed50.md)

4 new robots trained with our critic (Push, StackCube, YCB, Peg; seed 50) plus their standard-critic twins.
Method, labels and baselines fixed in advance; nothing changed after the results.

| Criterion | Result | Verdict |
| :-- | :-- | :-: |
| 1. Best average on the 4 new robots (all hallucinations) | ours 0.814 vs MOBILE 0.728, critic disagreement 0.704, dynamics ensemble 0.682, ELVIS 0.415 | **passed** |
| 2. Also for gradual and for sudden | gradual 0.822 (best rival 0.704); sudden 0.802 (best rival 0.741) | **passed** |
| 3. Never significantly worse than D or M on any robot | 0 of 4 worse; better on 2 of 4 (vs M) and 1 of 4 (vs D) | **passed** |
| 4. Bellman signals add at least 0.03 (all 8 robots) | 0.818 vs 0.758 without Bellman signals (+0.060) | **passed** |

All four pre-registered criteria passed. Exploratory: on sudden errors, our simpler readers (linear 0.840,
anchored residual alone 0.843) scored above the tree (0.802) on these 4 robots.

---

# Detection results per robot (judge: the simulator)

AUROC: how well a detector ranks real hallucinations above correct imagined steps (0.5 = guessing, 1.0 = perfect).
Best detector per robot in **bold**. \* = needs 4 extra world models; these exist only on robots trained with our critic.
Roles: practice = used to design the method (seed 10); new = trained afterwards, only tested (seed 40); extra = trained earlier, not used to design the tree reader (seeds 20, 30).

## All hallucinations

| Task | Seed | Critic | Role | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Push | 50 | ours |  | 0.896 | 0.842 | **0.936** | 0.848 | 0.524 | 0.810 | 0.210 |
| Push | 50 | normal |  | **0.928** | 0.886 | 0.901 | – | – | 0.754 | 0.187 |
| StackCube | 50 | ours |  | **0.882** | 0.874 | 0.866 | 0.775 | 0.833 | 0.738 | 0.201 |
| StackCube | 50 | normal |  | **0.815** | 0.768 | 0.795 | – | – | 0.751 | 0.414 |
| YCB | 50 | ours |  | 0.640 | **0.686** | 0.626 | 0.526 | 0.590 | 0.489 | 0.603 |
| YCB | 50 | normal |  | **0.821** | 0.804 | 0.745 | – | – | 0.724 | 0.414 |
| Peg | 50 | ours | extra | **0.836** | 0.813 | 0.769 | 0.764 | 0.780 | 0.779 | 0.644 |
| Peg | 50 | normal | extra | **0.729** | 0.604 | 0.600 | – | – | 0.506 | 0.500 |

**Averages (all hallucinations)**

| Group | Robots | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| All robots | 8 | **0.818** | 0.785 | 0.780 | 0.728 | 0.682 | 0.694 | 0.397 |
| Robots with our critic (like-for-like, all methods) | 4 | **0.814** | 0.804 | 0.799 | 0.728 | 0.682 | 0.704 | 0.415 |
| Push, robots with our critic | 1 | 0.896 | 0.842 | **0.936** | 0.848 | 0.524 | 0.810 | 0.210 |
| StackCube, robots with our critic | 1 | **0.882** | 0.874 | 0.866 | 0.775 | 0.833 | 0.738 | 0.201 |
| YCB, robots with our critic | 1 | 0.640 | **0.686** | 0.626 | 0.526 | 0.590 | 0.489 | 0.603 |
| Peg, robots with our critic | 1 | **0.836** | 0.813 | 0.769 | 0.764 | 0.780 | 0.779 | 0.644 |

## Gradual hallucinations

| Task | Seed | Critic | Role | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Push | 50 | ours |  | 0.895 | 0.803 | **0.935** | 0.780 | 0.536 | 0.726 | 0.248 |
| Push | 50 | normal |  | **0.933** | 0.874 | 0.912 | – | – | 0.760 | 0.196 |
| StackCube | 50 | ours |  | **0.870** | 0.854 | 0.849 | 0.786 | 0.833 | 0.771 | 0.200 |
| StackCube | 50 | normal |  | 0.790 | 0.755 | 0.744 | – | – | **0.803** | 0.358 |
| YCB | 50 | ours |  | **0.776** | 0.719 | 0.608 | 0.623 | 0.691 | 0.532 | 0.458 |
| YCB | 50 | normal |  | **0.851** | 0.801 | 0.740 | – | – | 0.751 | 0.387 |
| Peg | 50 | ours | extra | **0.749** | 0.681 | 0.625 | 0.627 | 0.638 | 0.637 | 0.384 |
| Peg | 50 | normal | extra | **0.703** | 0.591 | 0.551 | – | – | 0.493 | 0.495 |

**Averages (gradual hallucinations)**

| Group | Robots | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| All robots | 8 | **0.821** | 0.760 | 0.746 | 0.704 | 0.674 | 0.684 | 0.341 |
| Robots with our critic (like-for-like, all methods) | 4 | **0.822** | 0.764 | 0.754 | 0.704 | 0.674 | 0.667 | 0.323 |
| Push, robots with our critic | 1 | 0.895 | 0.803 | **0.935** | 0.780 | 0.536 | 0.726 | 0.248 |
| StackCube, robots with our critic | 1 | **0.870** | 0.854 | 0.849 | 0.786 | 0.833 | 0.771 | 0.200 |
| YCB, robots with our critic | 1 | **0.776** | 0.719 | 0.608 | 0.623 | 0.691 | 0.532 | 0.458 |
| Peg, robots with our critic | 1 | **0.749** | 0.681 | 0.625 | 0.627 | 0.638 | 0.637 | 0.384 |

## Sudden hallucinations

| Task | Seed | Critic | Role | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Push | 50 | ours |  | 0.896 | 0.875 | **0.937** | 0.904 | 0.515 | 0.879 | 0.179 |
| Push | 50 | normal |  | **0.917** | 0.914 | 0.876 | – | – | 0.741 | 0.165 |
| StackCube | 50 | ours |  | 0.905 | **0.909** | 0.898 | 0.756 | 0.834 | 0.677 | 0.204 |
| StackCube | 50 | normal |  | 0.861 | 0.794 | **0.891** | – | – | 0.654 | 0.522 |
| YCB | 50 | ours |  | 0.494 | 0.650 | 0.644 | 0.422 | 0.482 | 0.443 | **0.759** |
| YCB | 50 | normal |  | 0.691 | **0.818** | 0.765 | – | – | 0.614 | 0.524 |
| Peg | 50 | ours | extra | 0.912 | **0.927** | 0.893 | 0.884 | 0.903 | 0.902 | 0.870 |
| Peg | 50 | normal | extra | **0.812** | 0.644 | 0.755 | – | – | 0.547 | 0.516 |

**Averages (sudden hallucinations)**

| Group | Robots | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| All robots | 8 | 0.811 | 0.816 | **0.832** | 0.741 | 0.684 | 0.682 | 0.467 |
| Robots with our critic (like-for-like, all methods) | 4 | 0.802 | 0.840 | **0.843** | 0.741 | 0.684 | 0.725 | 0.503 |
| Push, robots with our critic | 1 | 0.896 | 0.875 | **0.937** | 0.904 | 0.515 | 0.879 | 0.179 |
| StackCube, robots with our critic | 1 | 0.905 | **0.909** | 0.898 | 0.756 | 0.834 | 0.677 | 0.204 |
| YCB, robots with our critic | 1 | 0.494 | 0.650 | 0.644 | 0.422 | 0.482 | 0.443 | **0.759** |
| Peg, robots with our critic | 1 | 0.912 | **0.927** | 0.893 | 0.884 | 0.903 | 0.902 | 0.870 |

## How often ours (tree) beats each method, robot by robot (robots with our critic)

| Compared with | All hallucinations | Gradual hallucinations | Sudden hallucinations |
| :-- | :-: | :-: | :-: |
| Ours (linear) | 3 of 4 | 4 of 4 | 1 of 4 |
| Ours: anchored only | 3 of 4 | 3 of 4 | 2 of 4 |
| MOBILE-style* | 4 of 4 | 4 of 4 | 3 of 4 |
| Dynamics ensemble* | 4 of 4 | 4 of 4 | 4 of 4 |
| Critic disagreement | 4 of 4 | 4 of 4 | 4 of 4 |
| ELVIS-style | 4 of 4 | 4 of 4 | 3 of 4 |

---

# Learned Bellman audit: ablations, data needs, transfer and statistics

8 robots (4 with our critic). Simulator labels; AUROC (0.5 guessing, 1.0 perfect). All audits use the same gradient-boosted tree settings as the main results.

## 1. Which signals matter (average over all robots)

| Audit reads… | All | Gradual | Sudden |
| :-- | :-: | :-: | :-: |
| Full audit (ours) | 0.818 | 0.821 | 0.811 |
| One-step δ only | 0.754 | 0.744 | 0.776 |
| Anchored δ only | 0.765 | 0.750 | 0.789 |
| One-step + anchored δ | 0.778 | 0.768 | 0.794 |
| All residuals, no head disagreement | 0.831 | 0.829 | 0.831 |
| All residuals, no multi-step δ | 0.819 | 0.820 | 0.814 |
| All residuals, no anchored δ | 0.815 | 0.815 | 0.813 |
| No Bellman: value, reward, step only | 0.758 | 0.747 | 0.755 |
| No Bellman: critic spread + value, reward, step | 0.791 | 0.788 | 0.786 |

## 2. How many episodes the audit needs (average over all robots, all hallucinations)

| Calibration episodes | AUROC |
| :-: | :-: |
| 5 | 0.760 |
| 10 | 0.792 |
| 15 | 0.788 |
| 25 | 0.818 |

## 3. Transfer: audit fitted on one robot, used on another (all hallucinations)

| Fitted on | Raw features | Standardised with the target's unlabeled data |
| :-- | :-: | :-: |
| the same robot | 0.818 (8 pairs) | 0.818 |
| another robot, same task | – (0 pairs) | – |
| a robot of another task | 0.739 (24 pairs) | 0.696 |

## 4a. Ours minus MOBILE-style (as published), per robot (paired bootstrap over episodes, all hallucinations)

| Robot | Ours | MOBILE-style | Difference | 95% interval | Significant? |
| :-- | :-: | :-: | :-: | :-: | :-: |
| real_seed_grounded_peg_s50 | 0.836 | 0.764 | +0.072 | [+0.015, +0.154] | yes |
| real_seed_grounded_push_s50 | 0.896 | 0.848 | +0.048 | [-0.031, +0.143] | no |
| real_seed_grounded_stack_s50 | 0.882 | 0.775 | +0.107 | [+0.004, +0.226] | yes |
| real_seed_grounded_ycb_s50 | 0.640 | 0.526 | +0.114 | [-0.093, +0.342] | no |

Ours significantly better on 2 of 4 robots, significantly worse on 0; mean difference +0.085.

## 4b. Ours minus Dynamics ensemble (as published), per robot (paired bootstrap over episodes, all hallucinations)

| Robot | Ours | Dynamics ensemble | Difference | 95% interval | Significant? |
| :-- | :-: | :-: | :-: | :-: | :-: |
| real_seed_grounded_peg_s50 | 0.836 | 0.780 | +0.056 | [-0.006, +0.157] | no |
| real_seed_grounded_push_s50 | 0.896 | 0.524 | +0.371 | [+0.129, +0.582] | yes |
| real_seed_grounded_stack_s50 | 0.882 | 0.833 | +0.049 | [-0.037, +0.156] | no |
| real_seed_grounded_ycb_s50 | 0.640 | 0.590 | +0.050 | [-0.109, +0.265] | no |

Ours significantly better on 1 of 4 robots, significantly worse on 0; mean difference +0.132.

## 5. Before vs after the task is already solved (robots with our critic, all hallucinations)

A window counts as 'after' if the real episode had already reached success at the window's start.

| Robot | Ours: before | Ours: after | Dynamics ensemble: before | Dynamics ensemble: after |
| :-- | :-: | :-: | :-: | :-: |
| real_seed_grounded_peg_s50 | 0.773 | 0.783 | 0.709 | 0.482 |
| real_seed_grounded_push_s50 | 0.679 | 0.916 | 0.567 | 0.511 |
| real_seed_grounded_stack_s50 | 0.834 | 0.845 | 0.768 | 0.862 |
| real_seed_grounded_ycb_s50 | 0.686 | 0.518 | 0.560 | 0.840 |

