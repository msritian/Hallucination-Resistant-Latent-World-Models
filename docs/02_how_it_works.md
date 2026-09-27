# 02 — How It Works, End to End

## Overview: training vs inference

```
TRAINING  (weights change)
   robot practices in the simulator → real experience stored in a replay buffer
   world model, critic, policy all learn together (one TD-MPC2 run)
   our change: the critic (and policy) learn ONLY from real states  ("grounding")

INFERENCE  (weights frozen)
   every 50 ms: plan with the world model; the checker audits every imagined step;
   plans are scored with trust; do the first action of the best plan
```

The **auditor itself is inference-only** (no training). We still have to train models because there are no ready-made TD-MPC2 models for ManiSkill3, and grounding changes how the critic is trained.

---

## Step 1 — Training (how the models are learned)

The robot acts in the simulator and stores every real step:

```
(real state s_t, action a_t, real reward r_t, real next state s_t+1)
```

Each update, TD-MPC2 trains all networks at once:

```
encoder:      real observation s_t          →  latent z_t  (512 numbers)
world model:  learn  d(z_t, a_t) ≈ z_t+1       (predict the next latent)
reward head:  learn  predicted reward ≈ r_t
critic Q:     learn  Q(z_t, a_t) ≈ r_t + 0.95 × Q(z_t+1, next action)
policy:       learn to pick actions the critic rates highly
```

### What "grounding" changes

| | Critic trains on | Risk |
| :--- | :--- | :--- |
| Stock TD-MPC2 | The world model's **imagined** latents | Learns to accept the model's mistakes |
| **Ours (grounded)** | Latents of **real** observations only | None of that; it only saw real physics |

It's about two lines of code in TD-MPC2's update. The policy is grounded the same way.

### How the critic learns what a state is worth

It learns step by step ("TD learning"), not by comparing with a goal:

```
target = reward now + 0.95 × Q(next state)
```

Value spreads backward from where reward is earned:

```
states:  A (far) → B → C → D (success, reward 1)
Q:       0.90      0.95  1.0
```

So Q ends up **higher the closer you are to success** — learned purely from rewards, no expert demos.

---

## Step 2 — Inference: what the planner does

Every 50 ms (20 Hz):

```
1. Encode the real observation → z0
2. Imagine 512 plans, each 12 actions long (24 of them suggested by the policy)
      world model: z0 →a0→ z1 →a1→ z2 → ... → z12   + imagined rewards r0 .. r11
3. SCORE every plan                     ← we change ONLY this step
4. Keep the best 64, sample new plans near them, re-score   (repeat 6 rounds)
5. Execute the first action of the best plan; throw the rest away
6. 50 ms later: start again from the new real state
```

---

## Step 3 — How the critic scores a state at inference

At inference the critic **compares against nothing**. It is a neural network forward pass; everything it "knows" is in its weights.

```
input:  [z_t (512 numbers), a_t (action numbers)]
  → layer → layer → 101 "bucket" scores
  → softmax → probability per bucket
  → expected bucket value → undo compression (symexp) → one number
do this for all 5 critic heads, average them  →  Q(z_t, a_t)
```

Meaning of the number: **"how much future reward I expect from here, taking this action."** It does not know whether z_t is real or imagined — it just evaluates whatever it's given.

---

## Step 4 — The checker (Bellman residual) on each imagined step

For each imagined step z_t → z_t+1:

```
delta_t = | Q(z_t, a_t)  −  ( r_t + 0.95 × Q(z_t+1, next action) ) |
```

- `Q(z_t, a_t)` — critic's opinion of this step
- `r_t` — the reward the world model predicted for this step
- `Q(z_t+1, next action)` — critic's opinion of the imagined next state (next action = policy's average action)

For real physics these match (small delta). For a hallucinated jump they don't (big delta).

### Example: a real step

```
Q(z_t, a_t) = 5.0,   r_t = 0.3,   Q(z_t+1) = 4.9
delta = | 5.0 − (0.3 + 0.95 × 4.9) | = | 5.0 − 4.96 | = 0.04     → believable
```

### Example: a teleport

```
Q(z_t, a_t) = 2.0,   r_t = 0,   Q(z_t+1) = 10.0     (next state looks like "at goal")
delta = | 2.0 − (0 + 0.95 × 10) | = | 2.0 − 9.5 | = 7.5        → impossible jump
```

### Compare with "normal"

Even correct predictions have some delta. So we compare with the normal delta for that step, `tau(t)`, measured on real replays where the model was known to be accurate:

```
s_t = delta_t / tau(t)

s ≤ 1  → as normal as a correct prediction
s = 2  → twice the normal error
```

---

## Step 5 — From suspicion to trust (the dial)

### Trust per step

```
w_t = 1                    if s_t ≤ 1
w_t = exp( −(s_t − 1) )    if s_t > 1
```

| s (suspicion) | w (trust) |
| :-: | :-: |
| ≤ 1 | 1.00 |
| 2 | 0.37 |
| 3 | 0.14 |
| 4 | 0.05 |

(Full version has a strictness knob kappa: `w = exp(−kappa × max(0, s − 1))`.)

### Remaining trust (Omega) — a battery that only drains

```
Omega_t = w_0 × w_1 × ... × w_t          (cumulative trust up to and including step t)
Omega_before_start = 1
```

Why multiply: each imagined state is built on the previous one, so one fake step makes everything after it fake too.

```
step:      0    1    2 (fake)   3
w:         1    1    0          1      ← step 3 looks fine alone...
Omega:     1    1    0          0      ← ...but the battery is already empty
```

Properties: always between 0 and 1, never goes up, once 0 stays 0.

---

## Step 6 — The score (trust-weighted λ-return)

```
Score = sum over steps t of  0.95^t × [ Omega_t × r_t  +  (Omega_before_t − Omega_t) × Q(z_t, a_t) ]
                                        ─────────────     ────────────────────────────────────────
                                           term 1                        term 2
        + 0.95^H × Omega_last × Q(z_H, policy action)
          ──────────────────────────────────────────
                          term 3
```

| Term | Meaning |
| :--- | :--- |
| 1 | Count each imagined reward only as much as it is still trusted |
| 2 | Wherever trust is **lost**, fill that share with the critic's realistic estimate at the state **before** the jump |
| 3 | The value after the plan ends, counted only if trust survived to the end |

Every share of the plan's future is counted exactly once: either from imagination (while trusted) or from the critic (where trust was lost).

### Worked example (from Research_WM.md, gamma = 1)

```
step:              0     1     2     3        4      5       end
imagined reward:   1     1     1     10       10     10      end value 10
                                     ↑ teleport (fake)
critic's realistic estimate at step 3:  Q(z3, a3) = 4
```

Only step 3 is suspicious; try three trust levels for it:

| Trust at step 3 | Term 1 | Term 2 | Term 3 | **Score** | Meaning |
| :-: | :-: | :-: | :-: | :-: | :--- |
| 1 | 1+1+1+10+10+10 = 33 | 0 | 1 × 10 = 10 | **43** | fooled (= normal TD-MPC2) |
| 0.5 | 3 + 0.5×30 = 18 | 0.5 × 4 = 2 | 0.5 × 10 = 5 | **25** | half-believed |
| 0 | 3 | 1 × 4 = 4 | 0 | **7** | caught (= cut at step 3, use critic) |

### Special cases (why this replaced the old three Options)

| Situation | Score becomes | Old equivalent |
| :--- | :--- | :--- |
| All steps trusted | Normal TD-MPC2 score | Option 1 (minus the penalty) |
| Trust lost fully at step k | Rewards before k + critic at k | Option 2 (cut) |
| Trust lost at step 0 | Q(z0, a0) — still ranks plans by first action | Option 3 (was −∞, could crash) |

Advantages: no cliffs (smooth), never crashes (no −∞), no extra penalty term needed.

---

## Step 7 — Calibration (how tau(t) is measured)

Done once per trained model, on episodes not used for training or testing:

```
1. Record 50 fresh real episodes
2. From every start point, replay the model open-loop along the REAL recorded actions
3. For each step, record delta_t AND the true error ||z_imagined − z_real||
4. Keep only replays whose prefix stayed accurate ("clean")
5. tau(t) = 95th percentile of delta_t on clean replays, separately for each step t
```

So `s ≤ 1` literally means "no bigger than a correct prediction's error at this step".

---

## Step 8 — Vector B: checking video world models (secondary)

### The problem

Researchers test robot policies inside **video world models** (e.g. NVIDIA Cosmos) instead of on real robots:

```
policy picks actions → video model generates the video → success detector watches → success / fail
```

Video models hallucinate too, e.g. the mug suddenly appears **in** the gripper. The success detector sees "mug in gripper" → success. Weak policies that trigger such glitches get inflated scores, so the **ranking of policies is wrong** ("visual optimism bias").

### The fix: the same Bellman check on video frames

```
1. Each frame → DINOv2 feature vector z_t (frozen image model)
2. Train on CLEAN videos:  Q_obs(z, a)  (expected future reward)  and  r_obs(z, a)  (step reward)
3. Residual per frame step, using the video's RECORDED next action a_t+1:
      delta_t = | Q_obs(z_t, a_t) − ( r_obs(z_t, a_t) + 0.95 × Q_obs(z_t+1, a_t+1) ) |
4. De-bias: subtract this policy's normal delta on its own clean videos
      delta_debiased_t = max(0, delta_t − policy's normal delta at step t)
   (an unusual-but-honest policy gets a slightly higher residual just because the critic
    was trained on other behavior — that shouldn't be punished)
5. Trust, as in Vector A:  s_t = delta_debiased_t / tau_video(t),  w_t = exp(−max(0, s_t − 1)),
                           Omega = w_0 × ... × w_last
6. Audited score of a policy = average over its videos of ( success_video × Omega_last )
```

### Example

```
Video 1 (honest):   pushes mug, mug arrives        success 1, Omega ≈ 1  → counts 1
Video 2 (glitch):   mug teleports into gripper     success 1, Omega ≈ 0  → counts 0
Video 3 (honest):   misses                         success 0             → counts 0
naive = 67%   audited = 33%
```

```
                 naive   audited   true (simulator)
Policy A (good)   70%      68%          70%
Policy B (weak)   75%      30%          28%
Policy C (ok)     50%      48%          50%
naive ranking B > A > C (wrong)     audited A > C > B (matches truth)
```

### How it is tested (controlled simulation study)

1. 10 PushCube policies of different quality; true success rate from 100 simulator episodes each.
2. Render their episodes as videos; insert "cube teleports to goal" glitches, **more often for weaker policies** (a video model that flatters bad policies).
3. Compare rankings vs the truth with **Spearman rank correlation** (1.0 = identical ranking).
4. Controls: **no glitches** (audited should equal naive), **same glitch rate for all**, **harmless glitches** (repeated frame, color jitter — should NOT be flagged; measures false alarms).

Why secondary: glitches are made by us and favor weak policies by design, so this demonstrates the mechanism but doesn't prove real video models behave this way.

## Step 9 — Source fix: training the world model to hallucinate less (secondary)

Vectors A and B **detect** hallucinations; the source fix tries to **prevent** them by adding delta as an extra training loss:

```
L_auditor  = (sum over 12 imagined steps of 0.95^t × delta_t) / (sum of 0.95^t) / scale(Q)
total loss = normal TD-MPC2 loss + ramp(step) × c × L_auditor
```

- Imagined rollouts follow **real recorded actions** from the replay buffer.
- `scale(Q)`: TD-MPC2's running estimate of typical Q size, so the loss doesn't depend on reward units.
- `c`: loss weight. `ramp`: 0 for the first 25% of training (critic still useless), then phased in over 10%.

### Only the world model may learn from this loss

| If this could learn from the loss… | …it could cheat like this |
| :--- | :--- |
| Reward head | Output `reward = Q(now) − 0.95 × Q(next)`, making delta = 0 without fixing anything |
| Critic | Change its values to match the hallucinations (the judge gets bribed) |
| Encoder | Reshape the latent space so fake jumps look consistent |
| Policy | Change the "next action" to make the numbers balance |

So only the **dynamics network** gets gradients from it.

### Two more guards

1. **Size cap:** rescaled so it never exceeds 10% of the main prediction loss (gradient direction kept).
2. **Cheating check** on held-out episodes:
   ```
   delta ↓  and real prediction error ↓   → real improvement
   delta ↓  and real prediction error ↑   → cheating the critic → reported as a failure
   ```

Why secondary: training models to agree with value estimates already exists (VAML, value equivalence, VIPO). First experiment to drop if compute runs short.
