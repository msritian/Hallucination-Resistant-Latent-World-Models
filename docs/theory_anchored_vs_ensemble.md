# Why anchored Bellman checks track compounding error and ensemble spread cannot

Two exact, elementary statements that explain the compounding results (docs/results_compounding_final.md). They give the
mechanism and a testable prediction. They are not deep theorems.

## Setup

A real start latent z_0 = enc(s_0) and actions a_0..a_t. The world model imagines ẑ_1, ẑ_2, … (ẑ_0 = z_0). The real
trajectory, encoded, is z_1, z_2, …. The reward head r̂(·,·), the critic Q, and V(z) = Q(z, μ_π(z)) are all the agent's
own networks. Write the (t+1)-step return of the model's own heads along a latent path x:

  G_t(x) = Σ_{k≤t} γ^k r̂(x_k, a_k) + γ^{t+1} V(x_{t+1})

The anchored residual (Signal Aa, `Rollout.anchored_residual`) is Aa_t = |Q(z_0, a_0) − G_t(ẑ)|.

## Proposition 1 (anchored residual = imagination error ± a dynamics-free floor)

Define
- the imagination error in value units: I_t = G_t(ẑ) − G_t(z). It is zero when the imagined latents equal the real
  ones, and it accumulates the reward and value consequences of every step of latent drift;
- the critic floor: C_t = Q(z_0, a_0) − G_t(z), the same anchored check computed on the REAL encoded trajectory. It does
  not involve the dynamics model at all.

Then Q(z_0,a_0) − G_t(ẑ) = C_t − I_t exactly, so

  | |I_t| − |C_t| |  ≤  Aa_t  ≤  |I_t| + |C_t|.

Consequences:
1. **Aa must grow when imagination error compounds.** Once |I_t| exceeds the floor |C_t|, Aa_t ≥ |I_t| − |C_t|. No model can
   hide a growing value-relevant drift from this check.
2. **Grounding matters.** C_t is the critic's own multi-step inconsistency on real data. A critic trained on real
   transitions (grounded) keeps it small. That is why detection is better with a grounded critic (H1b).
3. **False alarms come from the floor, and the floor is largest where value changes sharply.** |C_t| is large near
   contacts and value cliffs, even when the imagination is perfect (I_t = 0). This is the "decision-critical states"
   effect that made every avoid-flagged use fail (planning, offline pessimism), measured directly in the activity check.
4. **Only value-relevant error is seen.** Drift that changes neither predicted rewards nor values gives I_t = 0. That
   matches the PushCube NO-GO and the decision to label hallucinations by return error.

## Proposition 2 (ensemble spread is blind to shared error)

K dynamics heads, each rolled out on its own: ẑ^m_t, with errors e^m_t = ẑ^m_t − z_t. Let ē_t = mean_m e^m_t (the
error of the ensemble mean, the shared bias). Then, exactly,

  (1/K) Σ_m ‖e^m_t‖²  =  ‖ē_t‖²  +  (1/K) Σ_m ‖e^m_t − ē_t‖².
  (mean member error)   (shared bias)   (spread: what D, Dt, M, Mt measure)

Any alarm that is a function of the members' disagreement is unchanged if every member is moved by the same offset b.
It cannot distinguish an accurate ensemble from one whose members are all wrong in the same direction. Members trained on
the same data with the same architecture and the same encoder tend to extrapolate the same way, so as rollouts lengthen
the shared term can grow while the spread does not.

## Testable prediction and result (docs/results_trajectory_ensembles.md)

Prediction: with heads rolled out independently from the real start, the error of the ensemble mean grows faster than
the spread, so error / spread rises with depth and trajectory ensembles (Dt, Mt) under-react like D and M.

Result on 15 ManiSkill robots at 24 and 40 imagined steps: **not observed.** Spread grows faster than the shared error
(latent ×8.2 vs ×4.9, return ×80 vs ×35 at 40 steps); error / spread falls on 15/15 robots. Dt and Mt do grow with the
error (×6.6 and ×5.0). So Proposition 2 describes a possible failure, not the observed one, and the "ensembles go
blind" claim is restricted to one-step ensemble alarms (MOPO / MOBILE / MACURA-style), which stay flat. Proposition 1
still holds; the learned audit that builds on it ranks best at long range on average (significant at 24 steps,
12/15 and 11/15 robots vs Dt and Mt at 40).
