# Detection results per robot (judge: the simulator)

AUROC: how well a detector ranks real hallucinations above correct imagined steps (0.5 = guessing, 1.0 = perfect).
Best detector per robot in **bold**. \* = needs 4 extra world models; these exist only on robots trained with our critic.
Roles: practice = used to design the method (seed 10); new = trained afterwards, only tested (seed 40); extra = trained earlier, not used to design the tree reader (seeds 20, 30).

## Key takeaways (read this first)

- **22 robots** (4 tasks: Push, StackCube, YCB, Peg; several training runs each; our critic or the standard critic) and **7 detectors**, every number judged by the simulator. Published methods are used as published.
- **Overall, our tree reader is best in all three cases:** all 0.818, gradual 0.822, sudden 0.844 (averaged over all 22 robots). On the 11 robots where every method exists: 0.815 / 0.820 / 0.831 vs the best published ensemble method (MOBILE-style) 0.685 / 0.689 / 0.694.
- **1st overall on 9 of 11 robots**; beats MOBILE-style on 10 of 11, the dynamics ensemble on 9 of 11, critic disagreement and ELVIS-style on 11 of 11.
- **Per task** (robots with our critic, "all"): Push 0.934, StackCube 0.788, YCB 0.806, Peg 0.721. Best published method on every task; on Push the single anchored check is a hair higher (0.939), and on StackCube our linear reader is (0.805).
- **Where we lose:** two individual robots (StackCube s40, YCB s20), where the dynamics ensemble (4 extra models) was best. Results vary between robots for every method, so the averages are what count.

## What the "ours" columns mean

| Column | What it is |
| :-- | :-- |
| **Ours (tree)** | **Our main method:** the learned Bellman audit, a tree-based reader combining ~31 Bellman-check signals |
| Ours (linear) | The same ~31 signals, combined by a weighted sum instead of trees |
| Ours: anchored only | A single raw signal, the anchored Bellman residual (drift check), with no learned reader |

## Simple view: overall score, robot by robot

Robots trained with our critic, where every method exists ("all hallucinations"). The highest score per robot is in **bold**. \* = needs 4 extra world models.

| Robot | Ours (tree) | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style | Ours 1st? |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| Push s10 (practice) | **0.907** | 0.851 | 0.486 | 0.742 | 0.174 | ✅ |
| Push s40 (new) | **0.961** | 0.777 | 0.614 | 0.673 | 0.187 | ✅ |
| StackCube s10 (practice) | **0.793** | 0.736 | 0.764 | 0.751 | 0.320 | ✅ |
| StackCube s20 (extra) | **0.708** | 0.615 | 0.628 | 0.558 | 0.442 | ✅ |
| StackCube s30 (extra) | **0.898** | 0.827 | 0.812 | 0.803 | 0.297 | ✅ |
| StackCube s40 (new) | 0.754 | 0.797 | **0.812** | 0.753 | 0.286 | ❌ |
| YCB s10 (practice) | **0.775** | 0.663 | 0.712 | 0.590 | 0.456 | ✅ |
| YCB s20 (extra) | 0.732 | 0.571 | **0.791** | 0.484 | 0.558 | ❌ |
| YCB s30 (extra) | **0.884** | 0.473 | 0.461 | 0.461 | 0.839 | ✅ |
| YCB s40 (new) | **0.833** | 0.718 | 0.809 | 0.677 | 0.468 | ✅ |
| Peg s10 (extra) | **0.721** | 0.502 | 0.510 | 0.507 | 0.514 | ✅ |
| **Ours higher on** | | **10 of 11** | **9 of 11** | **11 of 11** | **11 of 11** | **9 of 11** |

**In one line:** across 4 tasks and 11 comparable robots, our detector is 1st overall on 9, and beats each published ensemble method on 9–10 of 11, with no extra models.

## All hallucinations

| Task | Seed | Critic | Role | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Push | 10 | ours | practice | 0.907 | 0.901 | **0.916** | 0.851 | 0.486 | 0.742 | 0.174 |
| Push | 10 | normal | practice | 0.884 | **0.934** | 0.933 | – | – | 0.739 | 0.141 |
| Push | 40 | ours | new | 0.961 | 0.953 | **0.962** | 0.777 | 0.614 | 0.673 | 0.187 |
| Push | 40 | normal | new | **0.888** | 0.846 | 0.862 | – | – | 0.823 | 0.143 |
| StackCube | 10 | ours | practice | 0.793 | **0.820** | 0.815 | 0.736 | 0.764 | 0.751 | 0.320 |
| StackCube | 10 | normal | practice | **0.866** | 0.774 | 0.730 | – | – | 0.708 | 0.344 |
| StackCube | 20 | ours | extra | 0.708 | **0.737** | 0.632 | 0.615 | 0.628 | 0.558 | 0.442 |
| StackCube | 20 | normal | extra | 0.741 | **0.834** | 0.726 | – | – | 0.725 | 0.191 |
| StackCube | 30 | ours | extra | **0.898** | 0.882 | 0.766 | 0.827 | 0.812 | 0.803 | 0.297 |
| StackCube | 30 | normal | extra | 0.782 | **0.804** | 0.678 | – | – | 0.689 | 0.122 |
| StackCube | 40 | ours | new | 0.754 | 0.783 | 0.749 | 0.797 | **0.812** | 0.753 | 0.286 |
| StackCube | 40 | normal | new | **0.925** | 0.880 | 0.784 | – | – | 0.818 | 0.292 |
| YCB | 10 | ours | practice | **0.775** | 0.766 | 0.614 | 0.663 | 0.712 | 0.590 | 0.456 |
| YCB | 10 | normal | practice | 0.783 | **0.836** | 0.644 | – | – | 0.485 | 0.745 |
| YCB | 20 | ours | extra | 0.732 | 0.650 | 0.591 | 0.571 | **0.791** | 0.484 | 0.558 |
| YCB | 20 | normal | extra | **0.774** | 0.749 | 0.566 | – | – | 0.488 | 0.702 |
| YCB | 30 | ours | extra | **0.884** | 0.805 | 0.617 | 0.473 | 0.461 | 0.461 | 0.839 |
| YCB | 30 | normal | extra | **0.788** | 0.680 | 0.737 | – | – | 0.599 | 0.738 |
| YCB | 40 | ours | new | **0.833** | 0.766 | 0.670 | 0.718 | 0.809 | 0.677 | 0.468 |
| YCB | 40 | normal | new | **0.838** | 0.735 | 0.672 | – | – | 0.581 | 0.629 |
| Peg | 10 | ours | extra | **0.721** | 0.615 | 0.563 | 0.502 | 0.510 | 0.507 | 0.514 |
| Peg | 10 | normal | extra | **0.763** | 0.598 | 0.544 | – | – | 0.567 | 0.490 |

**Averages (all hallucinations)**

| Group | Robots | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| All robots | 22 | **0.818** | 0.788 | 0.717 | 0.685 | 0.673 | 0.646 | 0.413 |
| Robots with our critic (like-for-like, all methods) | 11 | **0.815** | 0.789 | 0.718 | 0.685 | 0.673 | 0.636 | 0.413 |
| Push, robots with our critic | 2 | 0.934 | 0.927 | **0.939** | 0.814 | 0.550 | 0.707 | 0.181 |
| StackCube, robots with our critic | 4 | 0.788 | **0.805** | 0.740 | 0.744 | 0.754 | 0.716 | 0.336 |
| YCB, robots with our critic | 4 | **0.806** | 0.747 | 0.623 | 0.606 | 0.693 | 0.553 | 0.580 |
| Peg, robots with our critic | 1 | **0.721** | 0.615 | 0.563 | 0.502 | 0.510 | 0.507 | 0.514 |

## Gradual hallucinations

| Task | Seed | Critic | Role | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Push | 10 | ours | practice | **0.973** | 0.851 | 0.950 | 0.938 | 0.495 | 0.946 | 0.150 |
| Push | 10 | normal | practice | **0.977** | 0.962 | 0.959 | – | – | 0.794 | 0.142 |
| Push | 40 | ours | new | 0.971 | 0.982 | **0.982** | 0.814 | 0.634 | 0.749 | 0.179 |
| Push | 40 | normal | new | **0.883** | 0.824 | 0.873 | – | – | 0.806 | 0.145 |
| StackCube | 10 | ours | practice | 0.759 | **0.808** | 0.728 | 0.709 | 0.744 | 0.753 | 0.291 |
| StackCube | 10 | normal | practice | **0.862** | 0.753 | 0.717 | – | – | 0.739 | 0.286 |
| StackCube | 20 | ours | extra | 0.687 | **0.742** | 0.598 | 0.603 | 0.602 | 0.552 | 0.468 |
| StackCube | 20 | normal | extra | 0.742 | **0.833** | 0.717 | – | – | 0.739 | 0.173 |
| StackCube | 30 | ours | extra | **0.883** | 0.860 | 0.744 | 0.808 | 0.788 | 0.778 | 0.342 |
| StackCube | 30 | normal | extra | 0.751 | **0.775** | 0.639 | – | – | 0.669 | 0.118 |
| StackCube | 40 | ours | new | 0.730 | 0.766 | 0.728 | 0.765 | **0.780** | 0.721 | 0.310 |
| StackCube | 40 | normal | new | **0.906** | 0.856 | 0.776 | – | – | 0.797 | 0.267 |
| YCB | 10 | ours | practice | **0.855** | 0.786 | 0.675 | 0.749 | 0.826 | 0.671 | 0.371 |
| YCB | 10 | normal | practice | **0.804** | 0.780 | 0.777 | – | – | 0.526 | 0.576 |
| YCB | 20 | ours | extra | 0.726 | 0.705 | 0.633 | 0.560 | **0.801** | 0.494 | 0.502 |
| YCB | 20 | normal | extra | **0.792** | 0.693 | 0.692 | – | – | 0.542 | 0.603 |
| YCB | 30 | ours | extra | **0.894** | 0.796 | 0.549 | 0.429 | 0.415 | 0.462 | 0.842 |
| YCB | 30 | normal | extra | **0.766** | 0.644 | 0.724 | – | – | 0.596 | 0.711 |
| YCB | 40 | ours | new | **0.832** | 0.761 | 0.668 | 0.722 | 0.811 | 0.681 | 0.466 |
| YCB | 40 | normal | new | **0.837** | 0.731 | 0.673 | – | – | 0.577 | 0.622 |
| Peg | 10 | ours | extra | **0.709** | 0.591 | 0.543 | 0.483 | 0.494 | 0.483 | 0.501 |
| Peg | 10 | normal | extra | **0.743** | 0.565 | 0.512 | – | – | 0.533 | 0.449 |

**Averages (gradual hallucinations)**

| Group | Robots | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| All robots | 22 | **0.822** | 0.775 | 0.721 | 0.689 | 0.672 | 0.664 | 0.387 |
| Robots with our critic (like-for-like, all methods) | 11 | **0.820** | 0.786 | 0.709 | 0.689 | 0.672 | 0.663 | 0.402 |
| Push, robots with our critic | 2 | **0.972** | 0.916 | 0.966 | 0.876 | 0.565 | 0.847 | 0.164 |
| StackCube, robots with our critic | 4 | 0.765 | **0.794** | 0.699 | 0.721 | 0.728 | 0.701 | 0.352 |
| YCB, robots with our critic | 4 | **0.826** | 0.762 | 0.631 | 0.615 | 0.713 | 0.577 | 0.545 |
| Peg, robots with our critic | 1 | **0.709** | 0.591 | 0.543 | 0.483 | 0.494 | 0.483 | 0.501 |

## Sudden hallucinations

| Task | Seed | Critic | Role | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-- | :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Push | 10 | ours | practice | 0.898 | 0.908 | **0.912** | 0.840 | 0.485 | 0.715 | 0.177 |
| Push | 10 | normal | practice | 0.848 | 0.923 | **0.924** | – | – | 0.719 | 0.141 |
| Push | 40 | ours | new | **0.949** | 0.921 | 0.940 | 0.735 | 0.592 | 0.588 | 0.197 |
| Push | 40 | normal | new | **0.894** | 0.876 | 0.849 | – | – | 0.847 | 0.139 |
| StackCube | 10 | ours | practice | 0.823 | 0.830 | **0.891** | 0.760 | 0.782 | 0.749 | 0.345 |
| StackCube | 10 | normal | practice | **0.875** | 0.812 | 0.753 | – | – | 0.651 | 0.449 |
| StackCube | 20 | ours | extra | **0.818** | 0.712 | 0.808 | 0.680 | 0.763 | 0.593 | 0.306 |
| StackCube | 20 | normal | extra | 0.739 | **0.836** | 0.749 | – | – | 0.686 | 0.237 |
| StackCube | 30 | ours | extra | **0.930** | 0.928 | 0.815 | 0.868 | 0.864 | 0.857 | 0.201 |
| StackCube | 30 | normal | extra | 0.949 | **0.956** | 0.882 | – | – | 0.797 | 0.139 |
| StackCube | 40 | ours | new | 0.844 | 0.850 | 0.827 | 0.919 | **0.935** | 0.877 | 0.197 |
| StackCube | 40 | normal | new | **0.971** | 0.937 | 0.803 | – | – | 0.867 | 0.351 |
| YCB | 10 | ours | practice | 0.611 | **0.724** | 0.489 | 0.488 | 0.477 | 0.424 | 0.629 |
| YCB | 10 | normal | practice | 0.768 | **0.877** | 0.546 | – | – | 0.455 | 0.869 |
| YCB | 20 | ours | extra | 0.750 | 0.484 | 0.463 | 0.603 | **0.761** | 0.454 | 0.729 |
| YCB | 20 | normal | extra | 0.746 | 0.835 | 0.376 | – | – | 0.406 | **0.850** |
| YCB | 30 | ours | extra | **0.866** | 0.823 | 0.748 | 0.560 | 0.551 | 0.461 | 0.834 |
| YCB | 30 | normal | extra | **0.910** | 0.881 | 0.807 | – | – | 0.620 | 0.892 |
| YCB | 40 | ours | new | 0.859 | **0.908** | 0.718 | 0.575 | 0.737 | 0.561 | 0.533 |
| YCB | 40 | normal | new | **0.864** | 0.816 | 0.652 | – | – | 0.659 | 0.775 |
| Peg | 10 | ours | extra | **0.791** | 0.749 | 0.677 | 0.608 | 0.599 | 0.639 | 0.588 |
| Peg | 10 | normal | extra | **0.869** | 0.771 | 0.709 | – | – | 0.739 | 0.702 |

**Averages (sudden hallucinations)**

| Group | Robots | Ours (tree) | Ours (linear) | Ours: anchored only | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| All robots | 22 | **0.844** | 0.834 | 0.743 | 0.694 | 0.686 | 0.653 | 0.467 |
| Robots with our critic (like-for-like, all methods) | 11 | **0.831** | 0.803 | 0.753 | 0.694 | 0.686 | 0.629 | 0.431 |
| Push, robots with our critic | 2 | 0.924 | 0.914 | **0.926** | 0.787 | 0.539 | 0.652 | 0.187 |
| StackCube, robots with our critic | 4 | **0.854** | 0.830 | 0.835 | 0.807 | 0.836 | 0.769 | 0.262 |
| YCB, robots with our critic | 4 | **0.771** | 0.735 | 0.605 | 0.556 | 0.632 | 0.475 | 0.681 |
| Peg, robots with our critic | 1 | **0.791** | 0.749 | 0.677 | 0.608 | 0.599 | 0.639 | 0.588 |

## How often ours (tree) beats each method, robot by robot (robots with our critic)

| Compared with | All hallucinations | Gradual hallucinations | Sudden hallucinations |
| :-- | :-: | :-: | :-: |
| Ours (linear) | 8 of 11 | 7 of 11 | 6 of 11 |
| Ours: anchored only | 8 of 11 | 10 of 11 | 9 of 11 |
| MOBILE-style* | 10 of 11 | 10 of 11 | 10 of 11 |
| Dynamics ensemble* | 9 of 11 | 9 of 11 | 9 of 11 |
| Critic disagreement | 11 of 11 | 11 of 11 | 10 of 11 |
| ELVIS-style | 11 of 11 | 11 of 11 | 10 of 11 |
