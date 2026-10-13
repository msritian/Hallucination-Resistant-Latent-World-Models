# Compounding error: final results (Step 2)

Question ("Biased Dreams", 2026): as a latent rollout gets longer and its real error compounds, do hallucination alarms
grow with the error? Signal value on hallucinated windows, first 3 vs last 3 imagined steps; AUROC per step.
ManiSkill: 15 grounded robots (seeds 10-50) with dynamics ensembles D, M and critic spread B.
DMC: released TD-MPC2 MT30-48M, all 30 tasks (H=24) and 10 tasks (H=40); only critic-head spread B exists there.

## Summary

| Benchmark | H | True error growth | Ensembles (D / M / B) | Anchored Aa / Ac | Learned audit |
| :-- | --: | --: | :-- | :-- | :-- |
| ManiSkill, 15 robots | 24 | x18.1 | x0.92 / x1.11 / x1.92 (grow on 5, 4, 9 of 15) | x2.30 / x2.19 (15/15, 15/15) | x5.45 (15/15) |
| ManiSkill, 15 robots | 40 | x37.5 | x0.89 / x0.76 / x1.40 (grow on 6, 3, 6 of 15) | x2.64 / x2.66 (12/15, 15/15) | x14.0 (15/15) |
| DMC, 30 tasks | 24 | x36.2 | B x1.30 (18/30) | x12.4 / x9.35 (30/30, 30/30) | x3.25 (28/30) |
| DMC, 10 tasks | 40 | x112 | B x1.80 (6/10) | x26.4 / x18.7 (10/10, 10/10) | x7.17 (10/10) |

Per-unit mean growth ratios; counts = robots/tasks where the signal grows at all. Aa, Ac and the learned audit grow more
than D and M on 15/15 robots at both horizons (Aa vs M 14/15 at H24); more than critic spread B on 30/30 DMC tasks (Aa, Ac).
Ranking (per-step AUROC) is a separate question: the learned audit is the best detector at every step and horizon
(ManiSkill ~.81-.84, DMC ~.88-.92); raw anchored checks lose long-range ranking on ManiSkill (Aa .75 -> .65 at H40).
317M model: pending (first attempt ran out of GPU memory; resubmitted on >= 20 GB GPUs).

## Raw output: ManiSkill

```
######## H24, 15 robots
15 robots with ensembles. Mean true imagination error by step: 0.04 0.07 0.10 0.14 0.18 0.22 0.26 0.30 0.35 0.40 0.46 0.52 0.59 0.66 0.73 0.80 0.88 0.96 1.03 1.11 1.19 1.27 1.34

AUROC per imagined step (1 = first imagined transition):
signal k    1 k    2 k    3 k    4 k    5 k    6 k    7 k    8 k    9 k   10 k   11 k   12 k   13 k   14 k   15 k   16 k   17 k   18 k   19 k   20 k   21 k   22 k   23   first-3  last-3  change
LBA      0.79   0.84   0.84   0.83   0.82   0.82   0.82   0.82   0.82   0.82   0.82   0.83   0.82   0.83   0.84   0.85   0.85   0.85   0.85   0.85   0.84   0.84   0.84   0.83       0.84     +0.01
Aa       0.72   0.78   0.77   0.73   0.71   0.70   0.70   0.71   0.72   0.73   0.74   0.74   0.74   0.74   0.74   0.75   0.75   0.75   0.75   0.75   0.75   0.75   0.75   0.76       0.75     -0.01
Ac       0.72   0.78   0.77   0.74   0.71   0.70   0.70   0.69   0.68   0.67   0.66   0.66   0.65   0.64   0.64   0.64   0.64   0.64   0.64   0.63   0.63   0.63   0.63   0.76       0.63     -0.13
A        0.72   0.74   0.68   0.66   0.65   0.65   0.65   0.65   0.65   0.66   0.65   0.64   0.65   0.66   0.65   0.65   0.67   0.67   0.67   0.67   0.67   0.66   0.66   0.71       0.66     -0.05
D        0.70   0.73   0.72   0.69   0.68   0.68   0.67   0.66   0.66   0.67   0.68   0.68   0.69   0.69   0.69   0.68   0.69   0.70   0.70   0.71   0.71   0.70   0.71   0.72       0.71     -0.01
M        0.75   0.72   0.70   0.70   0.69   0.69   0.68   0.67   0.67   0.68   0.67   0.66   0.66   0.66   0.66   0.66   0.68   0.68   0.68   0.68   0.68   0.67   0.67   0.73       0.67     -0.05
B        0.66   0.61   0.61   0.62   0.62   0.62   0.64   0.64   0.65   0.65   0.65   0.64   0.65   0.64   0.65   0.66   0.68   0.68   0.68   0.68   0.68   0.66   0.67   0.63       0.67     +0.04

Growth from the first 3 to the last 3 imagined steps, on hallucinated windows: signal ratio vs error ratio
  LBA    signal x4.95   (true error x18.09)
  Aa     signal x2.14   (true error x18.09)
  Ac     signal x2.04   (true error x18.09)
  A      signal x0.48   (true error x18.09)
  D      signal x0.89   (true error x18.09)
  M      signal x0.61   (true error x18.09)
  B      signal x1.15   (true error x18.09)

Per-robot growth ratio (first 3 -> last 3 steps, hallucinated windows): mean [95% CI over robots]
  LBA    x5.45 [4.63, 6.29]   grows (>1) on 15/15 robots
  Aa     x2.30 [1.71, 2.99]   grows (>1) on 15/15 robots
  Ac     x2.19 [1.84, 2.61]   grows (>1) on 15/15 robots
  A      x0.55 [0.40, 0.71]   grows (>1) on 3/15 robots
  D      x0.92 [0.81, 1.03]   grows (>1) on 5/15 robots
  M      x1.11 [0.52, 1.95]   grows (>1) on 4/15 robots
  B      x1.92 [0.94, 3.48]   grows (>1) on 9/15 robots
  Aa grows more than D on 15/15 robots
  Ac grows more than D on 15/15 robots
  LBA grows more than D on 15/15 robots
  Aa grows more than M on 14/15 robots
  Ac grows more than M on 14/15 robots
  LBA grows more than M on 15/15 robots
  Aa grows more than B on 11/15 robots
  Ac grows more than B on 12/15 robots
  LBA grows more than B on 14/15 robots
######## H40, 15 robots
15 robots with ensembles. Mean true imagination error by step: 0.03 0.07 0.11 0.15 0.20 0.24 0.29 0.34 0.40 0.47 0.54 0.62 0.70 0.79 0.88 0.97 1.06 1.16 1.25 1.34 1.43 1.52 1.61 1.69 1.78 1.86 1.94 2.02 2.10 2.17 2.25 2.32 2.39 2.46 2.52 2.59 2.65 2.71 2.76

AUROC per imagined step (1 = first imagined transition):
signal k    1 k    2 k    3 k    4 k    5 k    6 k    7 k    8 k    9 k   10 k   11 k   12 k   13 k   14 k   15 k   16 k   17 k   18 k   19 k   20 k   21 k   22 k   23 k   24 k   25 k   26 k   27 k   28 k   29 k   30 k   31 k   32 k   33 k   34 k   35 k   36 k   37 k   38 k   39   first-3  last-3  change
LBA      0.79   0.81   0.81   0.79   0.79   0.79   0.79   0.79   0.79   0.79   0.80   0.78   0.79   0.81   0.83   0.82   0.82   0.82   0.81   0.80   0.82   0.79   0.81   0.80   0.79   0.79   0.76   0.76   0.78   0.81   0.80   0.82   0.81   0.81   0.81   0.81   0.81   0.81   0.81   0.81       0.81     +0.00
Aa       0.76   0.77   0.73   0.69   0.67   0.66   0.65   0.65   0.65   0.65   0.65   0.65   0.66   0.66   0.67   0.68   0.67   0.67   0.65   0.68   0.67   0.66   0.66   0.62   0.65   0.62   0.65   0.64   0.67   0.66   0.66   0.65   0.65   0.65   0.66   0.64   0.66   0.64   0.66   0.75       0.65     -0.10
Ac       0.76   0.78   0.76   0.70   0.67   0.66   0.63   0.60   0.59   0.58   0.57   0.57   0.56   0.58   0.58   0.59   0.60   0.60   0.60   0.57   0.60   0.59   0.60   0.58   0.60   0.59   0.59   0.61   0.61   0.63   0.61   0.59   0.61   0.62   0.61   0.62   0.62   0.61   0.61   0.77       0.62     -0.15
A        0.76   0.74   0.65   0.61   0.59   0.57   0.58   0.58   0.57   0.60   0.58   0.59   0.60   0.63   0.63   0.64   0.66   0.63   0.64   0.63   0.66   0.62   0.60   0.65   0.62   0.62   0.62   0.61   0.63   0.64   0.64   0.65   0.65   0.64   0.67   0.65   0.63   0.63   0.63   0.72       0.63     -0.08
D        0.72   0.77   0.74   0.70   0.68   0.66   0.65   0.64   0.64   0.64   0.65   0.65   0.66   0.68   0.69   0.69   0.69   0.69   0.70   0.72   0.74   0.73   0.72   0.72   0.72   0.74   0.75   0.72   0.72   0.74   0.76   0.74   0.74   0.73   0.74   0.74   0.72   0.73   0.72   0.74       0.72     -0.02
M        0.76   0.71   0.66   0.66   0.65   0.63   0.63   0.63   0.63   0.62   0.62   0.62   0.62   0.63   0.64   0.64   0.65   0.64   0.64   0.63   0.65   0.62   0.63   0.64   0.63   0.62   0.63   0.65   0.66   0.67   0.68   0.67   0.66   0.68   0.69   0.70   0.68   0.68   0.67   0.71       0.68     -0.03
B        0.58   0.55   0.53   0.54   0.54   0.54   0.56   0.56   0.58   0.59   0.58   0.59   0.60   0.60   0.62   0.63   0.65   0.63   0.64   0.63   0.65   0.62   0.64   0.60   0.62   0.62   0.63   0.64   0.65   0.65   0.67   0.67   0.66   0.68   0.69   0.68   0.66   0.66   0.65   0.55       0.66     +0.11

Growth from the first 3 to the last 3 imagined steps, on hallucinated windows: signal ratio vs error ratio
  LBA    signal x6.66   (true error x37.50)
  Aa     signal x1.92   (true error x37.50)
  Ac     signal x2.20   (true error x37.50)
  A      signal x0.32   (true error x37.50)
  D      signal x0.82   (true error x37.50)
  M      signal x0.39   (true error x37.50)
  B      signal x0.98   (true error x37.50)

Per-robot growth ratio (first 3 -> last 3 steps, hallucinated windows): mean [95% CI over robots]
  LBA    x13.99 [7.27, 22.58]   grows (>1) on 15/15 robots
  Aa     x2.64 [1.62, 4.03]   grows (>1) on 12/15 robots
  Ac     x2.66 [1.93, 3.51]   grows (>1) on 15/15 robots
  A      x0.47 [0.29, 0.67]   grows (>1) on 1/15 robots
  D      x0.89 [0.74, 1.04]   grows (>1) on 6/15 robots
  M      x0.76 [0.39, 1.16]   grows (>1) on 3/15 robots
  B      x1.40 [0.88, 1.96]   grows (>1) on 6/15 robots
  Aa grows more than D on 15/15 robots
  Ac grows more than D on 15/15 robots
  LBA grows more than D on 15/15 robots
  Aa grows more than M on 15/15 robots
  Ac grows more than M on 15/15 robots
  LBA grows more than M on 15/15 robots
  Aa grows more than B on 12/15 robots
  Ac grows more than B on 14/15 robots
  LBA grows more than B on 15/15 robots
```

## Raw output: DMC

```
30 tasks. True error growth first-3 -> last-3: x36.18
signal  growth(on hallucinated)  AUROC first-3  last-3
LBA     x2.17                 0.92         0.88
Aa      x10.33                 0.52         0.48
Ac      x6.99                 0.52         0.45
A       x0.99                 0.52         0.48
B       x1.04                 0.61         0.53
E       x0.98                 0.49         0.46

Per-task growth ratio (first 3 -> last 3 steps, hallucinated windows), 30 tasks: mean [95% CI]
  LBA    x3.25 [2.31, 4.48]   grows (>1) on 28/30 tasks
  Aa     x12.40 [9.25, 16.06]   grows (>1) on 30/30 tasks
  Ac     x9.35 [7.29, 11.80]   grows (>1) on 30/30 tasks
  A      x1.31 [1.03, 1.67]   grows (>1) on 15/30 tasks
  B      x1.30 [1.04, 1.59]   grows (>1) on 18/30 tasks
  E      x0.98 [0.97, 0.99]   grows (>1) on 7/30 tasks
  Aa grows more than B on 30/30 tasks
  Ac grows more than B on 30/30 tasks
  LBA grows more than B on 26/30 tasks

10 tasks. True error growth first-3 -> last-3: x112.45
signal  growth(on hallucinated)  AUROC first-3  last-3
LBA     x3.77                 0.88         0.87
Aa      x19.77                 0.47         0.53
Ac      x12.81                 0.48         0.51
A       x1.06                 0.47         0.51
B       x1.24                 0.60         0.54
E       x0.96                 0.50         0.46

Per-task growth ratio (first 3 -> last 3 steps, hallucinated windows), 10 tasks: mean [95% CI]
  LBA    x7.17 [3.70, 12.24]   grows (>1) on 10/10 tasks
  Aa     x26.39 [15.01, 40.19]   grows (>1) on 10/10 tasks
  Ac     x18.72 [10.29, 29.29]   grows (>1) on 10/10 tasks
  A      x1.52 [0.90, 2.35]   grows (>1) on 6/10 tasks
  B      x1.80 [0.94, 3.12]   grows (>1) on 6/10 tasks
  E      x0.96 [0.93, 0.98]   grows (>1) on 2/10 tasks
  Aa grows more than B on 10/10 tasks
  Ac grows more than B on 10/10 tasks
  LBA grows more than B on 9/10 tasks
```
