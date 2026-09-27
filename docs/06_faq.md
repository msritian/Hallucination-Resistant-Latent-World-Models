# 06 — FAQ

Short answers to questions that came up while designing the project.

### Do we train the world model, or is this only inference?
Both. The **auditor** is inference-only (no weights change). But we must **train** models first: there are no ready TD-MPC2 models for ManiSkill3, and grounding changes how the critic is trained. World model and critic train **together in one TD-MPC2 run** — no separate stage.

### What does "grounded critic" mean?
The critic learns only from **real** states and rewards from the simulator, never from the world model's imagined states. Stock TD-MPC2 trains the critic partly on imagined states, which could teach it to accept hallucinations.

### Does the critic compare a state to the goal or to a ground-truth trajectory?
No. At inference it's a neural network forward pass: 512 latent numbers + action in → one number out ("expected future reward from here"). What it knows is stored in its weights from training. During training it learned from the robot's own real attempts (not expert demos), using `target = reward + 0.95 × Q(next state)`.

### What exactly is "the error" the checker measures?
The Bellman residual:
```
delta_t = | Q(z_t, a_t) − ( r_t + 0.95 × Q(z_t+1, next action) ) |
```
Not just "Q at t minus Q at t+1" — it includes the step's reward and the discount.

### What is a "fake step's reward"?
The world model predicts a reward for every imagined step. If the step is hallucinated (cube teleports to goal), its predicted reward is fake too (e.g. 10 instead of the real ~1). Fake rewards are what make hallucinated plans look amazing.

### What is Omega / Omega_t?
Cumulative trust up to and including step t: `Omega_t = w_0 × w_1 × ... × w_t`. A battery that only drains. Once a step is fake, everything after it stays distrusted.

### Is "the critic's estimate where trust was lost" related to delta?
Same critic, same number, different job. `Q(z_t, a_t)` is the left side of delta (used to decide **how much** trust is lost) and is reused as the **replacement value** for the lost share. No extra computation.

### Why use the critic at the state *before* the jump?
`delta_t` checks the jump z_t → z_t+1. If it's fake, z_t+1 and everything after is suspect, but z_t is still believable. So we ask the critic at z_t.

### What is term 3 of the score?
The value after the plan ends: `gamma^H × Omega_last × Q(z_H, policy action)`. Counted only if trust survived to the end. If trust was lost earlier, that share was already paid in term 2.

### Is the trust-in-planning idea copied?
The **idea** of blending imagination and critic by trust exists (λ-return 1988, STEVE 2018, ELVIS 2026). We arrived at it independently but it is not novel. What's new is **how trust is decided** (the grounded Bellman residual).

### What's the "forward vs backward" thing?
Two ways to compute the same plan score: **forward** adds every step's contribution in one sum (Research_WM 3.2); **backward** starts at the end and folds in one step at a time (how ELVIS writes its formula). Same number either way. It is **not** a difference between ELVIS and us — we only write both backward to compare them.

### Is "trust only drains" a difference from ELVIS?
No (an earlier note got this wrong). In both methods, the weight on a later reward is a product of earlier weights, so both drain along the plan. The real structural difference is the **fallback**: when a step is distrusted, ELVIS still uses that step's imagined reward and the imagined next state's value; we use the critic's value from before the step. Those two fallbacks differ by exactly the Bellman residual.

### Why subtract each policy's "normal" residual in Vector B?
A policy that behaves unusually but honestly can get a slightly higher residual just because the video critic was trained on other behavior. Subtracting its residual on its own clean videos leaves only the extra residual caused by glitches.

### Why freeze everything except the dynamics network in the source fix?
Otherwise the model can make delta small by cheating: the reward head could output `Q(now) − 0.95 × Q(next)`, the critic could adjust to the hallucinations, the encoder could reshape latents. Only the part that predicts the next state should improve.

### Where did Research_WM's "Claims" and "Limitations" sections go?
Moved to `execution_final.md`: claims/hypotheses are Section 2, limitations are Section 10.

### How is ELVIS different, in one line?
ELVIS distrusts **states the critics disagree about** and falls back to values that still contain the fake step; we distrust **steps that break the Bellman equation** and fall back to the critic from before the step, with a critic trained on real data.

### Are ELVIS's critics trained on real data?
No — on imagined rollouts (only the start state is real). Ours are trained on real data only.

### Why does novelty depend on experiments?
Novel = nobody did it. Publishable = it's also **better**. ELVIS already blends imagination and critic; the only difference is the signal. If our signal isn't better than ELVIS's, the new idea adds nothing useful.

### What's the known blind spot?
A fake jump between two states with **equal value**: the Bellman equation still balances, delta stays small, the checker misses it. Measured explicitly (Type 3 in Phase 2).

### Why plan 12 steps instead of 3?
Many manipulation tasks need longer lookahead. TD-MPC2 stays at 3 because long imagination hallucinates. If trust stops hallucinations from fooling the planner, longer horizons become usable.

### Why 20 Hz and not 30 Hz?
ManiSkill3's native control rate is 20 Hz (50 ms per decision). The original doc assumed 30 Hz.

### Why gamma = 0.95, not 0.99?
TD-MPC2 sets the discount from episode length; for our 50–100-step tasks that gives 0.95.
