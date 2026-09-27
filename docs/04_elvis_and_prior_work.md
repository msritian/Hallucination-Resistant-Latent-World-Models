# 04 — ELVIS and Other Prior Work

Full sources are in [literature_review.md](literature_review.md).

## 1. ELVIS in one paragraph

**ELVIS** (arXiv 2605.04709, May 2026) plans with a learned world model like TD-MPC2, but further ahead (H = 15). It keeps several critic networks; where they **disagree** about a state (or rate it highly), it trusts the imagination less after that state and leans on the critics' average opinion instead. It reports first-or-second place on 14 DMC visual tasks vs TD-MPC2 and DreamerV3, and a real sand-spraying robot.

## 2. What ELVIS and we share

- Same planning loop (MPPI): imagine many plans → score → keep the best → repeat.
- Same goal: plan further ahead without an ensemble of world models.
- Same **kind** of scoring: a dial that blends "believe the imagination" with "use the critic".

The blending formula itself is classic RL (the λ-return, Sutton 1988), also used by STEVE (2018). **Neither ELVIS nor we invented it.**

## 3. Full side-by-side

| | ELVIS | Ours |
| :--- | :--- | :--- |
| Base system | New (Dreamer-style world model + TD-MPC2-style planning) | TD-MPC2, almost unchanged |
| World model | RSSM: memory + random latent | TD-MPC2 deterministic latent |
| Critics | M separate **V(state)** networks | TD-MPC2's 5 **Q(state, action)** heads |
| Critics trained on | **Imagined** rollouts (only the start state is real) | **Real** data only (grounded) |
| What decides trust | Do critics **disagree about a state** (plus its value level)? | Is the **step** from one state to the next consistent (Bellman residual)? |
| Fallback when a step is distrusted | Imagined reward of the step + critics' value of the **next** (possibly fake) state — still contains the hallucination | Critic's value from **before** the step — hallucination-free |
| Calibration | Running normalization | Measured on real replays (true model error) |
| Planner | Multi-mode MPPI | Standard TD-MPC2 MPPI |
| Tasks | DMC visual + real sand spraying | ManiSkill3 manipulation (state input) |

## 4. Why "multiple critics" works for ELVIS

Several critics with the same design start from different random weights. On familiar states they all learn the same answer (**agree**); on unfamiliar states each guesses differently (**disagree**). Disagreement = "never seen anything like this".

## 5. The two formulas compared

### Forward vs backward is NOT a difference

A plan's score can be computed **forward** (add every step's contribution in one sum — how Research_WM 3.2 writes ours) or **backward** (start at the end, fold in one step at a time — how ELVIS writes its score). Same number either way, like adding a list left-to-right or right-to-left. We write both backward only to compare them side by side.

### Both, written backward (G_t = score of the plan from step t to the end)

```
ELVIS:  G_t = r_t + gamma × [ (1 − lambda_t) × V(z_t+1)  +  lambda_t × G_t+1 ]

OURS:   G_t = (1 − w_t) × Q(z_t, a_t)  +  w_t × [ r_t + gamma × G_t+1 ]

start of the backward pass:  G_H = critic's value of the final imagined state;   score = G_0
```

| Symbol | Meaning |
| :--- | :--- |
| `lambda_t` | ELVIS's weight at step t, from how much its critics disagree about state z_t |
| `w_t` | Our trust in step t, from the step's Bellman residual |
| `V(z_t+1)` | Critics' average value of the next state (in TD-MPC2 terms: `Q(z_t+1, policy action)`) |
| `Q(z_t, a_t)` | Critic's value from step t, before the step happens |

Trusted share (weight = 1): both give `r_t + gamma × G_t+1` — the normal sum. **Identical.**

Distrusted share (weight = 0) — the real difference:

```
ELVIS falls back to:   r_t + gamma × V(z_t+1)     ← still uses the step's imagined reward AND imagined next state
OURS  falls back to:   Q(z_t, a_t)                ← critic's view from BEFORE the step
```

For a real step these are equal (that's the Bellman equation). For a fake step they differ by **exactly the Bellman residual**:

```
| Q(z_t, a_t) − ( r_t + gamma × V(z_t+1) ) |  =  delta_t
```

So when ELVIS distrusts a step, its fallback still contains that step's hallucination; ours does not.

### Correction: "trust only drains" is NOT a difference

In both methods the weight on a later reward is a **product** of the earlier weights (ELVIS: lambda_0 × lambda_1 × …; ours: w_0 × w_1 × … = Omega). Both drain along the plan. Earlier notes listing "ELVIS sets λ fresh, ours only drains" as a difference were wrong.

### Corrected list of real differences

1. **Signal** (main): critic disagreement about a state vs Bellman residual of a step.
2. **Fallback**: imagined reward + next-state value (contains the hallucination) vs critic from before the step.
3. **Critic training**: imagined rollouts vs real data only.
4. **Thresholds**: running normalization vs calibrated against real model error.

## 6. Same plan through both (numbers invented to show the mechanics)

```
step:     0      1      2         3          end
state:   z0 --> z1 --> z2 --> z3(FAKE) --> z4(FAKE)
reward:   1      1      10        10         end value 10
                        ↑ teleport happens during step 2 (z2 → z3)
critic, realistic:  Q at z0 = 5,  z1 = 4,  z2 = 3
honest plan scores about 5;   normal TD-MPC2 scores 1+1+10+10+10 = 32
```

### Ours

```
delta at step 2 = |3 − (10 + 10)| = 17 → trust 0
Omega:  1, 1, 0, 0

score = 1 + 1 + (lost 100%) × Q(z2) = 1 + 1 + 3 = 5      → not fooled
```

### ELVIS, best case (it notices the fake state z3: dial = 0 there)

```
dials:   z0 = 1, z1 = 1, z2 = 1, z3 = 0

backward:
  end      critics(z4) = 10
  step 3   10 + critics(z4) 10           = 20    (dial 0 at z3 → stop imagining after step 3)
  step 2   10 + 20                       = 30    (dial 1 at z2 → z2 looks normal)
  step 1   1 + 30                        = 31
  step 0   1 + 31                        = 32    → still fooled (≈25 if critics rate z4 lower)
```

Why: ELVIS judges **states**, so it only reacts once it **reaches** the fake state — the reward of the jump into it (10 at step 2) is already counted, the fake state's own reward is always added, and its fallback reads the critics at the fake next state.

## 7. Where each should win (expectations, not results)

| Situation | ELVIS | Ours |
| :--- | :--- | :--- |
| Fake jump to a **familiar high-value** state (teleport to goal) | Weak: critics agree | **Strong**: value jumps with no reward |
| Fake jump to a **weird unfamiliar** state | **Good**: critics disagree | Probably good |
| Fake jump between **equal-value** states | Maybe (if critics disagree) | **Blind** |
| **Real but new** states | May wrongly distrust | Fine if the step is consistent |
| Critic learned from hallucinations | Risk (trained on imagination) | Avoided (grounded) |

This is why we also test **Signal C = ours + critic disagreement**.

## 8. Other closest prior work

| Work | What it does | Difference from us |
| :--- | :--- | :--- |
| **STEVE** (2018) | Blends different imagination lengths by ensemble uncertainty | Needs dynamics + reward + critic ensembles; used in training, not planning |
| **M2AC** (2020) | Masks model rollouts where dynamics-ensemble disagreement is high | Ensemble; training |
| **AdaMVE** (2019) | Learns a separate model-error predictor to pick horizon | Separate network; training |
| **DMVE** (2020) | Picks horizon by image reconstruction error | Needs a decoder |
| **MACURA**, adaptive truncation (2026) | Truncate rollouts at high ensemble uncertainty | Ensemble; training |
| **MOBILE** (2023) | Penalizes offline RL by Bellman-target inconsistency across dynamics ensemble | Closest Bellman-based signal; needs ensemble; offline training |
| **VIPO** (2025) | Regularizes model training by value inconsistency | Close to our source fix |
| **LOOP** (2021) | Planner + critic trained model-free on real data | Precedent for a real-data critic in planning; doesn't audit steps |
| **TD-M(PC)²** (2025) | Fixes TD-MPC2 value overestimation | Related to how grounding changes value learning |
| **Operator-on-F** (2026) | Checkpoint diagnostic on TD-MPC2 | Warning: raw Bellman residual weakly predicted checkpoint quality (different use) |
| **Hallucination in WMs is Predictable** (2026) | Tokenizer/seed-variance signals | Not critic-based; used for data collection |
| **HaWMPO** (2026) | Learned hallucination score down-weights rewards for VLA training | Close to Vector B's "success × trust" idea |
| **Scalable Policy Eval with Video WMs** (2025) | Evaluates policies in video models with a VLM judge | Notes hallucination bias, doesn't correct it (Vector B's target) |

## 9. Novelty summary

| Element | Novel? |
| :--- | :-: |
| Trust-weighted horizons / blending with critic | No (STEVE, ELVIS, λ-return) |
| Our scoring formula specifically | Low (design variations of ELVIS's) |
| **Critic's Bellman residual on each imagined step as the planning trust signal** | **Yes (not found in prior work)** |
| **Grounding as what makes the residual valid** | **Yes, as a tested claim** |
| Calibration against true model error | Moderate |
| Source fix | Low |
| Vector B correction | Moderate (demo only) |

Correct way to present the score in the paper: *"a λ-return with step-dependent λ (Sutton 1988; as in STEVE and ELVIS), where λ is set from the calibrated Bellman residual of a grounded critic."*
