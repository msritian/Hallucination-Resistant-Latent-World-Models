# Is detection easy only because weak models make huge, obvious errors?

Tool: `python -m src.tools.difficulty --success success.json feat_*.tar.gz` on the main H=12 dumps: 11 ManiSkill robots ×
2 critics = 22 models (Push, StackCube, YCB, Peg; seeds 10–40). Success = mean of the last 3 training evaluations.
Hallucination bar eps_hall = P99 of the model's own one-step errors (reward units); ratio = error / eps_hall.

## 1. Detection does not rely on huge errors

AUROC of hallucinations of a given size vs all correct steps (stratified by imagined step), mean over 22 models:

| signal | barely past the bar (1–2×) | 2–4× | huge (>4×) | all |
| :-- | :-- | :-- | :-- | :-- |
| LBA (ours) | **0.812** | 0.834 | 0.865 | 0.819 |
| Aa | 0.713 | 0.724 | 0.767 | 0.717 |
| M (MOBILE-style) | 0.690 | 0.689 | 0.720 | 0.685 |
| D (ensemble) | 0.678 | 0.680 | 0.721 | 0.673 |
| B (critic spread) | 0.648 | 0.651 | 0.707 | 0.646 |
| E (ELVIS-style) | 0.420 | 0.402 | 0.367 | 0.413 |

Huge errors are slightly easier for every signal (+0.05 for LBA), but LBA keeps 0.81 on the hardest, near-threshold
cases, and its lead over the best rival (~0.12) is the same in every bin.

## 2. Weak agents do not make bigger errors here, and detection is not better on them

| group | models | eps_hall (reward units) | median error ratio | share >4× | LBA all | LBA 1–2× |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| strong (success ≥ 0.7) | 10 | 0.61 | 2.43 | 22% | **0.833** | 0.822 |
| weak (success < 0.7) | 12 | 0.38 | 1.91 | 15% | 0.807 | 0.804 |

Spearman over the 22 models: success vs LBA AUROC +0.34 (better agent → better detection), eps_hall vs LBA AUROC +0.51.

Reading: the concern predicts weak agents → bigger, more obvious errors → higher AUROC. The data show the opposite on
every link: weak agents' hallucinations are closer to the bar, not further, and detection is slightly worse on them.
Together with DMC (official fully trained checkpoints, 0.90), under-training makes the headline numbers conservative.

Caveat: absolute error sizes differ by task because reward scales differ (Push hits large rewards at success), so the
comparison uses each model's own bar; per-task strong vs weak is confounded with task (YCB/Peg are the weak ones).
