# 05 — Experiments, Ablations, and Outcomes

Details and exact settings are in `execution_final.md`.

## 1. Order of work (phases)

| Phase | What | Why |
| :-: | :--- | :--- |
| 0 | Build: ManiSkill3 wrapper, grounded TD-MPC2, extra dynamics heads, audited planner, calibrator, tests | Everything must pass tests before training |
| 1 | Pilot training on PushCube (seed 10): stock model + grounded model, 200k steps | Checkpoints for Phase 2; measure training speed |
| 2 | **Calibration + go/no-go detection test** | Cheap check: does the signal work at all, and beat ELVIS's? |
| 3 | Pilot planning evaluation on PushCube | First look at whether long-horizon planning improves |
| 4 | Full training: 3 tasks × 5 seeds | Main results |
| 5 | Source fix training (secondary) | Only if compute allows |
| 6 | Analysis, tables, figures | |
| 7 | Vector B (video demo) | Secondary |

## 2. The go/no-go test (Phase 2)

### What we do

Take imagined replays of real episodes and **plant fake steps** at step 3, plus use the model's **real mistakes**:

| Type | Fake | Tests |
| :-: | :--- | :--- |
| 1 | Random noise (kept valid for TD-MPC2's latent format) | Weird states |
| 2 | Blend toward a goal state, strength 10%–100% | "Teleport to goal" (our target case) |
| 3 | Swap in a real state with the **same value** | Our blind spot |
| 4 | No planting: the model's own real errors, labeled by true latent error | Natural hallucinations |

### What we measure

**AUROC**: how well a signal separates fake steps from real ones (1.0 = perfect, 0.5 = coin flip). For signals A (ours), B (critic spread), C (ours + spread), D (PETS ensemble), E (ELVIS), M (MOBILE), on both grounded and stock critics.

### Decision rule

```
CE = best of the critic-ensemble signals = max(AUROC of B, AUROC of E)

A signal QUALIFIES if (grounded model):
   AUROC ≥ 0.80 on Type 2 (strength ≥ 50%) and on Type 4
   AND AUROC on Type 4 ≥ CE + 0.05

GO           → A qualifies                     → use A in the planner
GO (combined)→ A doesn't, C does               → use C
NO-GO        → neither                         → stop and rethink before Phase 4
```

Always report Type 3 (blind spot), grounded vs stock (evidence for grounding), and D/M.

### PushCube result and the amended gate (2026-09-27)

- **PushCube: NO-GO** under the rule above (ours 0.71 on teleports, 0.68 on natural errors; critic spread 0.65; ensemble 0.95 on natural errors).
- **Diagnostic:** only 37% of the natural "mistakes" change value. On **value-changing** mistakes: ours **0.80**, critic spread 0.73, ensemble **0.60**. Grounding lowered critic noise (0.044 vs 0.071).
- **Amended gate for the harder tasks** (written down before their results): same rule, but natural errors are labeled by **value error** (Type 4v). The original gate is still reported. If both harder tasks fail it, the planning-method framing is dropped.

## 3. Training runs

All runs collect data with the **normal H = 3 planner**. Every comparison is a planner swap at test time on the same trained model.

| Run | Training | Size |
| :-: | :--- | :--- |
| 1 | Stock TD-MPC2 | 3 tasks × 5 seeds |
| 2 | Grounded + 4 extra dynamics heads (heads don't affect the main model) | 3 × 5 |
| 3 | Grounded + source-fix loss | 3 × 5 |

Tasks: PushCube (200k steps), PickSingleYCB (1M), PegInsertionSide (1M; replaced by StackCube if the baseline scores < 10%, decided before any audited results).

## 4. Arms (what is compared)

| Arm | Model | H | Score | Purpose |
| :-: | :-: | :-: | :--- | :--- |
| 0 | Run 1 | 3 | Stock TD-MPC2 planner untouched | Sanity: our planner code matches stock |
| 1 | Run 1 | 3 | Standard | Default baseline |
| 2 | Run 1 | 12 | Standard | Does long horizon hurt? |
| 3b | Run 2 | 3 | Standard | Cost of grounding |
| 3 | Run 2 | 12 | Standard | Grounded, no auditor |
| 4 | Run 2 | 12 | Trust score with PETS signal | vs dynamics ensemble |
| 4b | Run 2 (+ Run 1) | 12 | **ELVIS-style score** | **vs closest prior work** |
| **5** | Run 2 | 12 | **Trust score with our signal (A or C)** | **Our method** |
| 6b | Run 3 | 12 | Standard | Source fix alone |
| 6 | Run 3 | 12 | Trust score | Source fix + auditor |

## 5. Ablations

| Ablation | Question it answers |
| :--- | :--- |
| **Horizon sweep** H = 3, 6, 12, 24 for Arms 3, 4b, 5 | Headline figure: does standard planning degrade with H, and does ours stay flat? |
| **Signal** A, B, C, D, E, M in the same trust score | Which signal is best? |
| **Signal vs structure**: ELVIS signal in our score vs Arm 4b; our signal vs ELVIS signal in our score | Is our gain from the **signal** or the **formula**? |
| **Trust strictness** kappa = 0.5, 1, 2, ∞ (∞ = hard cut) | Does soft trust beat hard cutting? |
| **Calibration**: clean-filtered tau(t) vs unfiltered vs single global tau | Does calibration matter? |
| **Grounding**: auditor on stock (Run 1) models | Is grounding necessary for the signal? |

**Fair tuning:** ours and ELVIS each get at most 12 configurations, tuned only on the PushCube pilot with separate seeds, then frozen.

## 6. Measurements

- **Success rate**, 50 episodes per (task, seed, arm), same episode seeds for every arm (paired comparison).
- **IQM** across tasks × seeds with bootstrap 95% confidence intervals.
- **Paired differences** per seed: Arm 5 − Arm 3, **Arm 5 − Arm 4b (key)**, Arm 5 − Arm 4.
- **Effective horizon** `H_eff = sum of Omega`, how often trust hits ~0.
- **Latency**: planner time at H = 3, 12, 24; target auditor overhead < 3 ms within the 50 ms budget.
- **Source fix:** open-loop prediction error; failure if error rises > 10% while delta falls (gaming).

## 7. Hypotheses (pre-registered)

| ID | Claim |
| :-: | :--- |
| H1 | Grounded residual detects hallucinations (AUROC ≥ 0.80) and beats critic-ensemble signals |
| H2 | Standard TD-MPC2 degrades as H grows (if not at 12, test 16 and 24) |
| H3 | Our trust score keeps long-horizon planning ≥ the H = 3 baseline |
| H4 | Ours ≥ ELVIS-style and ≥ dynamics ensemble, at lower cost than the ensemble |
| H5 | (secondary) Source fix reduces model error without gaming |
| H6 | (secondary) Vector B restores the true policy ranking |

## 8. Possible outcomes

| Outcome | What we can publish |
| :--- | :--- |
| Phase 2 GO + Arm 5 beats 4b and 3 at long horizons | Main method paper |
| Phase 2 GO, but planning gains are small | Detection/analysis paper ("a free, calibrated hallucination detector") |
| Signal ties ELVIS but is cheaper than ensembles | Weaker; workshop |
| Phase 2 NO-GO | Rethink before spending compute; possibly an analysis of when the residual fails |
