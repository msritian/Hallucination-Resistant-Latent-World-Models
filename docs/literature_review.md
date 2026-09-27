# Literature Review & Novelty Assessment

**Date:** 2026-09-26
**Scope:** Prior work relevant to [`Research_WM.md`](../Research_WM.md) and [`execution_final.md`](../execution_final.md). This is a focused review from targeted searches, not an exhaustive survey; re-run the searches before submission, because this area moves month to month.

## Table of Contents

1. [Bottom Line](#1-bottom-line)
2. [Closest Work: ELVIS (2026)](#2-closest-work-elvis-2026)
3. [Related Work by Theme](#3-related-work-by-theme)
4. [What Is and Isn't Novel](#4-what-is-and-isnt-novel)
5. [Consequences for the Plan](#5-consequences-for-the-plan)
6. [References](#6-references)

---

## 1. Bottom Line

- **Adaptive, uncertainty-gated horizons are a crowded area** (MVE, STEVE, M2AC, AdaMVE, DMVE, MACURA, ELVIS, 2026 truncation and conformal-horizon papers). We cannot claim "trust the imagined future only as far as it is reliable" as our idea.
- **Almost all prior methods get reliability from ensembles** (dynamics, reward, or critic), from reconstruction error, or from a separately learned error model.
- **As far as this review found, no published method uses the Bellman residual of the agent's own critic, evaluated on each imagined transition, as a per-step trust signal for test-time planning.** Nor does any pair such a signal with a critic trained only on real states so that the residual is a valid check.
- **ELVIS (May 2026) is the closest work.** Its planning score, a λ-return with a per-step λ driven by critic-ensemble uncertainty, overlaps heavily with our trust-weighted λ-return. Our planning score is therefore not a contribution on its own; the **signal** and the **grounding** are.
- **The source fix overlaps with value-aware model learning** (VAML, value equivalence, VIPO). Its novelty is low; it should be positioned as secondary.
- **Vector B's setting is active** (video-world-model policy evaluation, hallucination-aware weighting). A Bellman-consistency de-biasing of such evaluations appears unexplored, but ours is only a sim-to-sim demonstration.

The paper therefore stands or falls on one empirical question: **does the grounded critic's Bellman residual detect hallucinated steps better than critic-ensemble spread (ELVIS-style) and dynamics-ensemble disagreement?** The Phase 2 gate in `execution_final.md` tests exactly this.

---

## 2. Closest Work: ELVIS (2026)

**ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC** (arXiv 2605.04709, May 2026).

### 2.1 What ELVIS Does

| Component | ELVIS |
| :--- | :--- |
| World model | Dreamer-style RSSM (deterministic memory + stochastic latent); **no dynamics ensemble** |
| Planner | Gaussian-mixture MPPI (several modes), TD-MPC2-style latent planning, H = 15 (ablation: H = 15 ≫ H = 5) |
| Uncertainty | Ensemble of M latent **state-value** critics: mean $\mu_t$, std $\sigma_t$, $\text{UCB}_t = \mu_t + \beta\sigma_t$ |
| Per-step λ | $\lambda_t = \lambda_{\max} - (\lambda_{\max} - \lambda_{\min}) \cdot \text{norm}(\text{UCB}_t)$, with norm = running mean/variance with clipping, mapped to [0, 1] |
| Path score | $G_t = \hat r_t + \gamma\big((1-\lambda_t)\,\mu_{t+1} + \lambda_t G_{t+1}\big)$, with $G_{H-1} = \mu_{H-1}$ |
| Critic training | On **imagined** rollouts, with the same λ-return as target |
| Benchmarks | 14 DMC visual tasks (first or second on every task vs TD-MPC2, DreamerV3); zero-shot real sand-spraying task |
| Key ablation | Fixed λ / no UCB gives a "large drop in aggregated score" |
| Stated limits | Dense-reward continuous control only; extra compute from long-horizon rollouts; no latency analysis |

### 2.2 Overlap With Us

- The same mechanism: a λ-return that moves weight from look-ahead onto the critic where uncertainty is high.
- The same goal: longer planning horizons than TD-MPC2 without a dynamics ensemble.
- ELVIS's signal is essentially our **Signal B** (critic-ensemble spread), plus the value level itself through the UCB.

### 2.3 Differences From Us

| | ELVIS | Ours |
| :--- | :--- | :--- |
| Signal | Critic disagreement about a **state** (plus its value level) | Bellman **consistency of a transition** |
| Critic | Trained on imagined rollouts | **Grounded**: trained only on real states |
| Calibration | Running normalization of UCB | Per-step thresholds calibrated against **true latent error** on held-out real episodes |
| Fallback when distrusting a step | Imagined reward + value of the next (possibly fake) state | Critic's value from before the step (differs from ELVIS's fallback by exactly the Bellman residual) |
| High value | Higher UCB → **less** look-ahead (value level lowers λ) | Value level alone never lowers trust |
| Domain | DMC visual locomotion + one real task | ManiSkill3 state-based manipulation |
| Extras | Gaussian-mixture MPPI, recurrent memory | Video audit (Vector B), source fix, detection study |

Note on the value level: because $\text{UCB} = \mu + \beta\sigma$, ELVIS shortens look-ahead in states the critic merely rates highly, not only in uncertain ones. That coupling is a concrete, testable point of difference.

---

## 3. Related Work by Theme

### 3.1 Uncertainty-Gated Rollout Horizons

Our planning score is algebraically a λ-return with step-dependent λ (λ_t = w_t). λ-returns are due to Sutton (1988); adapting λ per step or state is established (e.g., White & White, 2016). The rows below apply this idea to model-based RL.

| Work | Signal | Used for | Relation to us |
| :--- | :--- | :--- | :--- |
| λ-returns (Sutton, 1988); step-dependent λ (White & White, 2016) | — | Blending multi-step returns with bootstrapped values | The mathematical form of our score |
| MVE (Feinberg et al., 2018) | None (fixed H) | Value targets | Origin of model-based value expansion |
| **STEVE** (Buckman et al., 2018) | Variance across dynamics, reward, and Q ensembles | Value targets (training) | Origin of reliability-weighted horizon blending; needs ensembles; training-time |
| **M2AC** (Pan et al., 2020) | Dynamics-ensemble disagreement | Masking model rollouts (training) | "Trust the model when it is confident"; up to ~20-step rollouts; ensemble |
| AdaMVE (Xiao et al., 2019) | Learned multi-step model-error function | Adaptive horizon for value targets | A separately learned error predictor, not a critic residual |
| DMVE (Wang et al., 2020) | Image reconstruction error | Adaptive horizon | Needs a decoder |
| MACURA | Model uncertainty | Truncating rollouts | Ensemble-based |
| **ELVIS** (2026) | Critic-ensemble UCB | λ-return for **planning** | Closest; see Section 2 |
| Adaptive Rollout Truncation (arXiv 2609.21482, 2026) | Dynamics-ensemble or MC-dropout variance, calibrated threshold | Truncating offline world-model training rollouts | Calibrated truncation; ensemble signal; training-time |
| Conformal Orbit-Valid Trust Horizons (arXiv 2606.24946, 2026) | One-step latent residual + split-conformal calibration | Certifying trust horizons | Calibrated trust horizons; needs equivariance; certification, not planning |
| Adaptive-horizon TD-MPC2 variants (e.g., endovascular navigation, arXiv 2608.18647) | MPPI proposal dispersion | Per-step horizon choice | A different, planner-internal signal |

### 3.2 Bellman / Value Consistency as a Signal

| Work | What it measures | Used for | Relation to us |
| :--- | :--- | :--- | :--- |
| **MOBILE** (Sun et al., ICML 2023) | Inconsistency of Bellman **targets** across a **dynamics ensemble** | Reward penalty in offline model-based RL | Closest Bellman-based signal; needs an ensemble; offline training, not planning |
| VIPO (2025) | Gap between data-derived and model-derived value estimates | Regularizer in model training | Close to our **source fix** |
| Operator-on-F diagnostic (arXiv 2607.04464, 2026) | k-step latent operator error on observables; compares to Bellman residual | Ranking TD-MPC2 checkpoints | **Warning:** unnormalized Bellman residual correlated only weakly with return at checkpoint level (Spearman −0.10). This is a different use (checkpoint ranking, not per-step detection), but our preflight must show the per-step case works |
| TD-uncertainty for exploration (Flennerhag et al., 2020) | Distribution over TD errors | Exploration bonus | Different purpose |
| "Why Should I Trust You, Bellman?" (2022) | Bellman error vs value error | Analysis | Bellman error can be a poor proxy for value error; supports using calibration, not raw residual |

### 3.3 Critics Trained on Real Data, Used in Planning

| Work | Relation to us |
| :--- | :--- |
| **LOOP** (Sikchi et al., CoRL 2021) | H-step lookahead with a terminal value learned by a **model-free** off-policy critic on real data; analyzes the model-error vs value-error trade-off. Precedent for a real-data critic inside a planner; does not use it to audit transitions. |
| TD-M(PC)² (Lin et al., 2025) | Shows TD-MPC2 value overestimation from planner/policy mismatch; fixes it with a policy constraint. Relevant because grounding also changes TD-MPC2's value learning. A possible complementary fix, and a reason to measure grounding's cost (Arm 3b). |

**Implication:** "grounding" is not a new concept. Our claim is narrower: grounding is what makes the critic's residual a valid consistency check (tested by the grounded vs ungrounded detection comparison).

### 3.4 Value-Aware Model Learning (Relevant to the Source Fix)

| Work | Relation to us |
| :--- | :--- |
| VAML (Farahmand et al., 2017) | Train the model to induce the same Bellman operator as the real environment. |
| Value Equivalence Principle (Grimm et al., 2020, 2022) | Characterizes models that are value-equivalent to the environment. |
| Value-Consistent Representation Learning (2022) | Value consistency as a representation objective. |
| VIPO (2025) | Value-inconsistency loss added to model training. |

**Implication:** training the dynamics to reduce Bellman inconsistency is established. Our source fix differs mainly in its anti-gaming controls and gradient isolation, which are engineering and analysis rather than a new idea.

### 3.5 Hallucination in World Models

| Work | Signal | Relation to us |
| :--- | :--- | :--- |
| Hallucination in World Models is Predictable and Preventable (arXiv 2606.27326, 2026) | Tokenizer round-trip residual, flow instability, inter-seed variance (Dreamer 4) | Detection for data collection, not planning; no critic signal |
| World-Coherent Decoding (arXiv 2609.02159, 2026) | Forward-model coherence of candidate trajectories | Test-time self-verification without a value function |
| Imagined Rollouts are Kinematic, Not Dynamic (arXiv 2607.05966, 2026) | Diagnosis on TD-MPC2 and Dreamer | Explains why long rollouts fail; motivation for us |
| HaWMPO (arXiv 2609.09941, 2026) | Supervised hallucination model from DINOv3 similarity, depth, optical flow, image quality | Rewards scaled by $(1 - \alpha H)$ during VLA policy optimization; conceptually close to Vector B's "success × trust" |
| Dual-Frontier (arXiv 2609.26293, 2026) | Bellman-residual identity for certified decision trust | LLM tool-use agents; theory-heavy; different setting |

### 3.6 Video World Models for Policy Evaluation (Vector B)

| Work | Relation to us |
| :--- | :--- |
| Scalable Policy Evaluation with Video World Models (arXiv 2511.11520) | Cosmos-Predict2 + action conditioning, VLM judge; Pearson 0.83–0.88 on synthetic tasks; notes hallucinations bias evaluation but does not correct for them. The problem Vector B targets. |
| SC3-Eval (arXiv 2606.18610), RoboWorld (arXiv 2607.01060) | Recent video-model policy-evaluation frameworks. |
| How Should World Models Be Evaluated for Embodied Decision-Making? (arXiv 2606.15032) | Position paper: evaluation should track decisions, not realism. Supports our framing. |

---

## 4. What Is and Isn't Novel

| Element | Novelty | Reason |
| :--- | :-: | :--- |
| Reliability-weighted horizons / fallback to critic | ✗ | STEVE, M2AC, ELVIS, and others |
| Trust-weighted λ-return as a planning score | Low | ELVIS uses the same λ-return mechanism; our fallback choice and calibration are design differences, not a contribution on its own |
| **Critic's Bellman residual on imagined transitions as a per-step planning trust signal, with no extra models** | **✓ (not found)** | Nearest: MOBILE (ensemble target inconsistency, offline training), ELVIS (critic-ensemble spread) |
| **Grounding as what makes the residual valid** | **✓ as a claim** | The concept exists (LOOP); the claim that it is needed for detection is new and testable |
| Per-step thresholds calibrated against true latent error | Moderate | Calibrated truncation exists (2609.21482, 2606.24946); calibrating a critic residual on ground-truth-filtered replays is new in detail |
| Source fix | Low | VAML, value equivalence, VIPO |
| Vector B: Bellman-consistency de-biasing of video evaluation | Moderate | Problem well known; this correction appears unexplored; sim-to-sim only |

**Recommended contribution statement:**

> A critic trained only on real transitions turns its own Bellman residual into a free, calibrated detector of hallucinated imagined steps. As a trust signal for long-horizon latent planning, it needs no extra models, and we test whether it outperforms critic-ensemble (ELVIS-style) and dynamics-ensemble signals.

---

## 5. Consequences for the Plan

Applied in `execution_final.md` and `Research_WM.md`:

1. **ELVIS-style baseline arm (Arm 4b):** ELVIS's λ-return and UCB rule, using TD-MPC2's 5 critic heads, in our planner, with the same tuning budget as our method.
2. **Two more detection signals in the preflight:** ELVIS's UCB (Signal E) and a MOBILE-style ensemble Bellman-target spread (Signal M, free from the Run 2 auxiliary heads).
3. **Stricter gate:** the residual must beat the best critic-ensemble signal, max(B, E), not just B.
4. **Signal-vs-structure ablation:** Signal B inside our trust structure vs Arm 4b separates "better signal" from "better score structure".
5. **Source fix repositioned as secondary**, with VAML, value-equivalence and VIPO cited.
6. **Literature table and novelty statement** in `Research_WM.md` updated to match this review.

---

## 6. References

- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1). (λ-returns)
- White, M. & White, A. (2016). A greedy approach to adapting the trace parameter for temporal difference learning. *AAMAS*. (step-dependent λ)
- [TD-MPC2 (Hansen et al., 2024)](https://arxiv.org/abs/2310.16828)
- [ELVIS (arXiv 2605.04709)](https://arxiv.org/abs/2605.04709)
- [STEVE (Buckman et al., 2018)](https://arxiv.org/abs/1807.01675)
- [M2AC (Pan et al., 2020)](https://arxiv.org/abs/2010.04893)
- [AdaMVE: Learning to Combat Compounding-Error in MBRL](https://arxiv.org/abs/1912.11206)
- [DMVE (Wang et al., 2020)](https://arxiv.org/abs/2009.09593)
- [MACURA (summary)](https://lacuna.tiptreesystems.com/direction/uncertainty-aware-adaptive-planning-horizons/txn_2bab06139f6b455284506a63fcb3a621)
- [Adaptive Rollout Truncation Based on Epistemic Uncertainty (arXiv 2609.21482)](https://arxiv.org/html/2609.21482)
- [Conformal Orbit-Valid Trust Horizons (arXiv 2606.24946)](https://arxiv.org/abs/2606.24946)
- [Progressive Experience Fusion for Multi-Task World Model Control (arXiv 2608.18647)](https://arxiv.org/pdf/2608.18647)
- [MOBILE: Model-Bellman Inconsistency (ICML 2023)](https://proceedings.mlr.press/v202/sun23q.html)
- [VIPO (arXiv 2504.11944)](https://arxiv.org/html/2504.11944v3)
- [Operator-on-F planning-time diagnostic (arXiv 2607.04464)](https://arxiv.org/abs/2607.04464)
- [Temporal Difference Uncertainties as a Signal for Exploration](https://arxiv.org/abs/2010.02255)
- [Why Should I Trust You, Bellman?](https://arxiv.org/abs/2201.12417)
- [LOOP: Learning Off-Policy with Online Planning](https://arxiv.org/abs/2008.10066)
- [TD-M(PC)² (arXiv 2502.03550)](https://arxiv.org/abs/2502.03550)
- [Value Equivalence Principle (Grimm et al., 2020)](https://arxiv.org/abs/2011.03506)
- [Calibrated Value-Aware Model Learning (arXiv 2505.22772)](https://arxiv.org/html/2505.22772v1)
- [Value-Consistent Representation Learning](https://arxiv.org/abs/2206.12542)
- [Hallucination in World Models is Predictable and Preventable (arXiv 2606.27326)](https://arxiv.org/html/2606.27326v1)
- [World-Coherent Decoding (arXiv 2609.02159)](https://arxiv.org/abs/2609.02159)
- [Imagined Rollouts are Kinematic, Not Dynamic (arXiv 2607.05966)](https://arxiv.org/abs/2607.05966)
- [HaWMPO (arXiv 2609.09941)](https://arxiv.org/html/2609.09941)
- [Dual-Frontier (arXiv 2609.26293)](https://arxiv.org/html/2609.26293)
- [Scalable Policy Evaluation with Video World Models (arXiv 2511.11520)](https://arxiv.org/html/2511.11520)
- [SC3-Eval (arXiv 2606.18610)](https://arxiv.org/abs/2606.18610)
- [RoboWorld (arXiv 2607.01060)](https://arxiv.org/abs/2607.01060)
- [How Should World Models Be Evaluated for Embodied Decision-Making? (arXiv 2606.15032)](https://arxiv.org/abs/2606.15032)
