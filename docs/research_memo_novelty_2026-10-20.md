# Where can the Bellman audit still be novel? (research memo, 2026-10-20)

## The fact that reframes the search

In the run-time warning study (docs/monitor_results.md, 11 robots) a PERFECT warning (oracle, 25% of decisions)
raised success only from 0.580 to 0.648, and a random warning already gave 0.615. In the planner study
(docs/detplan_results.md) a fixed lambda = 0.8 gave as much as audit-driven lambda. Value expansion (powered, interim)
shows no arm beating plain training. Conclusion: in a policy that replans every step, hallucinations are cheap, so even
perfect detection has little to buy. The detector is not the bottleneck; the setting is.
Independent support: ARC-Bench (arXiv 2609.05461) shows closed-loop replanning masks broken action ranking in JEPA world
models; success collapses when replanning is reduced.

## Landscape check (Oct 2026)

| Area | Recent work | Signal | Gap for us |
| :-- | :-- | :-- | :-- |
| Label-free latent hallucination detection | MEND (2609.39182, DICTA 2026): score network on real transitions, AUROC <= 0.80, 2 navigation envs, one-step only | learned density/score | no value grounding, no compounding, no control use |
| Where to refuse imagination | DreamLedger (2608.23863): credit from settled execution history | history | needs past executions in similar conditions |
| Adaptive open-loop execution (world action models) | FFDC (2605.06222), CheckVLA (2607.26789), TempoWAM (2608.09492), Continue-or-Replan (2608.03483), DREAM-Chunk (2606.18589) | learned verifiers / progress monitors / real-vs-imagined checks | all need a trained verifier or real observations; none is training-free or value-grounded |
| Adaptive imagination budget | RISE (2608.20430): learned latent evaluator, driving | learned | needs labelled counterfactual data |
| Video-WM policy evaluation | HaWMPO, EchoArena, StressDream, Cosmos Policy | learned hallucination scorers | crowded; we have no video model |

## Candidate directions

| # | Direction | Novelty | Cost | Risk |
| :-- | :-- | :-- | :-- | :-- |
| 1 | **Commitment study: value of detection grows with open-loop commitment.** Execute k planned actions before replanning; gate k with the audit decided BEFORE acting; oracle bound; baselines D, Mt, prediction-error (future-vs-reality) gate, random at matched planning calls | High as a combined result: training-free, value-grounded gate + a measured link between commitment length and detection value; ties compounding (audit grows, ensembles flat) to a downstream outcome | eval only, existing robots | the real-observation check (prediction error) may dominate once the robot sees each step |
| 2 | Value-of-detection theory: downstream gain <= P(hallucination) x cost of acting on it, which scales with commitment k | Medium; frames all null results as predicted | paper only | must match #1's data |
| 3 | Audit-directed data collection (explore where plans hallucinate, vs Plan2Explore disagreement) | Medium | new training runs | TD-MPC2 learning proved insensitive (VE) |
| 4 | Task-level trust for generalist world models (MT30/MT80 on held-out tasks) | Medium-high, unexplored | eval on released checkpoints | critic not grounded on unseen tasks |
| 5 | Analysis paper as is: accurate, compounding-robust, training-robust detection; replanning absorbs its value | Medium (cf. ARC-Bench) | none | weaker at a main venue |

## Recommendation

Pilot #1 first (cheap, uses existing robots and harness), with #2 as its framing. Pre-register before running.
Note for positioning: the learned audit uses simulator labels on 25 calibration episodes; MEND is label-free. Report
the label-free anchored residual (Aa) alongside, and state the label cost explicitly.
