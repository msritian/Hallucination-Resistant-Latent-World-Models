# Training-maturity study: results (pre-registered in docs/preregistration_maturity.md)

Question: does the detector look good only because under-trained world models make big, obvious errors? Here the task
is held fixed and only the amount of training changes: 9 new grounded robots (PushCube 200k, StackCube 500k,
PickSingleYCB 1M steps; seeds 60-62), each evaluated at 10, 25, 50, 75 and 100% of training with exactly the main
protocol (50 new episodes per snapshot, detector fitted on 25 and scored on 25, H = 12, simulator labels with each
snapshot's own thresholds). All 45 snapshots had enough labels; no exclusions.
Tool: `python -m src.tools.maturity mfeat_*.tar.gz`.

## Pre-registered criteria (EARLY = 25%, LATE = 100%; 95% bootstrap CI over the 9 robots)

| | Criterion | Result | Verdict |
| :-- | :-- | :-- | :-: |
| M1 | Detection improves with training | learned-audit AUROC +0.070 [+0.021, +0.114], higher on 7/9 robots | **PASS** |
| M2 | Not inflated by immature models | CI lower bound +0.021 > −0.03 | **PASS** |
| M3 | Mechanism: the critic floor shrinks | relative floor \|C\| lower at 100% on 8/9 robots (need 7) | **PASS** |
| M4 | Sanity: the agents improve | success +0.420 [+0.182, +0.662] | reported |

## Means over the 9 robots

| Training | 10% | 25% | 50% | 75% | 100% |
| :-- | :-: | :-: | :-: | :-: | :-: |
| Agent success | 0.26 | 0.34 | 0.61 | 0.72 | 0.76 |
| Learned audit (LBA) AUROC | 0.819 | 0.792 | 0.832 | 0.843 | 0.862 |
| Anchored residual (Aa) AUROC | 0.713 | 0.642 | 0.725 | 0.738 | 0.774 |
| Ensemble D AUROC (5 heads) | 0.627 | 0.616 | 0.667 | 0.674 | 0.710 |
| Critic floor \|C\| / value spread | 0.052 | 0.053 | 0.046 | 0.024 | 0.024 |
| Median \|I\| on hallucinations / median \|C\| | 4.8 | 6.4 | 6.8 | 10.8 | 26.6 |

Spearman(agent success, LBA AUROC) over all 45 (robot, snapshot) points: +0.40.

## Reading

- Detection gets better, not worse, as the same robot trains: under-training makes the headline numbers conservative.
- The mechanism is the one Proposition 1 predicts (Aa = |C − I|): the grounded critic's error floor C halves between
  25% and 100%, so the imagination error I stands out more against it (signal-to-floor ×4).
- The learned audit is ahead of the 5-head ensemble at every stage (by 0.15-0.19 AUROC).

## Caveats (reported, not hidden)

- The 10% snapshot (report-only by pre-registration) scores 0.819, above the 25% snapshot. On StackCube and YCB the
  10% models have ~0 success; some of their errors are easy to catch. From 25% onward the trend is upward.
- Not every robot rises: YCB s60 (0.863 → 0.802) and YCB s62 (0.828 → 0.784) end lower; both are weak agents
  (32% and 38% success at 100%). The average trend holds; individual robots vary.
