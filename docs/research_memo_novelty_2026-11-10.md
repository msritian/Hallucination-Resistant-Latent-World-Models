# Novelty memo, 2026-11-10 (after the commitment study and the powered VE study)

## Where we stand

- Detection: strong (LBA 0.81-0.92 step-stratified AUROC; improves with training; robust to compounding).
- Control: every downstream use is null (planner, warning light, VE x2, commitment). In the commitment study even a
  perfect detector of our label (return error) gave +0.03 over random; Aa and PE gave about +0.055 (exploratory).
- Hypothesis now under test (docs/preregistration_commitment_confirm.md): the errors that matter for control are those
  that change the decision, not those that make the imagined value wrong.

## New literature (Jun-Nov 2026) and what it means for us

| Paper | What it does | Overlap / gap |
| :-- | :-- | :-- |
| Hansen & Wang, "Hallucination in World Models is Predictable and Preventable" (2606.27326) | Label-free hallucination predictors for a 350M Dreamer-4-style video world model (round-trip residual, flow instability, inter-seed variance); AUROC 0.87-0.94; used for curiosity data collection | Closest competitor, by the TD-MPC2 author. Visual/latent signals only, no value or Bellman signal, no run-time control use. Our value-grounded audit is complementary; we must cite and position against it. |
| Wang, Fan, Huang, "Not All Errors Matter: Decision-Relevant Prediction Error" (2609.32322) | Offline evaluation metric: error on decision-relevant state dims predicts planning success (rho -0.84 vs -0.25) | Supports our hypothesis. Offline, gridworld, needs known relevance. Gap: a run-time, before-acting estimator in a real agent (our DBA). |
| "Rethinking World Models for Safety-Critical Embodied Systems" (2609.03774) | Position: uncertainty should be tied to whether it changes the action ranking | Calls for exactly what DBA / decision oracles measure; no method. |
| "Operator-on-F" (2607.04464) | For TD-MPC2, reward error and Bellman residual correlate weakly with return; operator error tracks it | Independent evidence that value/Bellman error is not the control-relevant quantity; consistent with our null results. |
| "Planners Exploit Deterministic Error" (2611.11780) | TD-MPC2 planning collapses from errors that persist across planner iterations and are shared across candidates, not from large errors | Another "which errors matter" result; suggests a ranking-stability signal. |
| ARC-Bench (2609.05461), pico-WAM (2611.01039), AD-WM (2609.30264), TRAP (2605.01950) | Action-ranking failures are masked by replanning; ranking stability matters | Supports the commitment setting and ranking-based signals. |
| Model-Bellman Inconsistency (ICML 2023), In-Distribution Imagination (2609.38673), GIRL (2604.07426) | Bellman-estimate disagreement across models; rollout error accumulates while ensemble disagreement stays flat; grounding to the current observation | Prior art for Bellman-based uncertainty (ensemble-based) and for anchoring; none is a critic-anchored, training-free detector. |

**Reading.** The field is converging on "which errors matter for decisions" in 2026, but mostly as offline
metrics or position papers. A run-time, before-acting detector of decision-relevant errors in a standard agent,
with a pre-registered causal test, is still open. That is the gap to target, and fast.

## Candidate directions (ranked)

1. **Confirmation study (running).** Decision oracles (act, reg) and the decision-aware audit (DBA) on 12 fresh robots.
2. **Decision-flip signal, training-free.** TD-MPC2 already has 5 critic heads and 5 dynamics heads. Rescore the
   planner's final candidate pool under each head and measure how often the chosen plan stops being the winner
   (rank stability). This is a direct, label-free estimate of "would the action change", in the spirit of 2609.03774
   and 2611.11780. Cheap: evaluation only, reuses the commitment harness. Run as a separate pre-registered study
   after the confirmation results, so as not to change a pre-registered design.
3. **Position the detection result against Hansen & Wang.** Add their label-free signal types that transfer to
   TD-MPC2 (TD-MPC2 has no decoder, but action-sensitivity, i.e. how much the imagined values change when the
   actions are shuffled, transfers directly) as baselines in the detection tables. Analysis only.
4. **Audit-directed exploration.** Hansen & Wang show hallucination signals help as curiosity for data collection.
   Untested for value-grounded signals. Needs training runs (expensive); only if 1-2 give a positive result.
5. **Write-up.** Regardless of 1-2: the detection results plus the decision-relevance analysis are a coherent paper;
   the strongest version adds a positive control result from 1 or 2.

Dropped (tested 2026-11-10 on existing data): early-episode failure prediction. Mean AUROC 0.58, reversed on YCB;
the critic's value predicts failure better.
