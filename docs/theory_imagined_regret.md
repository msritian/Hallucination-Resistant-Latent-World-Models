# Why imagined regret is the right thing to gate on (theory note)

Notation (plain text): s = real state, z = latent state, Q = learned critic (mean over critic heads), pi = learned
policy, Q* = true action value. A plan is a = (a_0, ..., a_{H-1}) from the real state s_0. Committing k actions means
a_1 .. a_{k-1} are executed blind, at real states s_1 .. s_{k-1} the robot never looks at.

## 1. What committing costs

Executing the stale plan action a_j at the real state s_j, instead of replanning, costs the regret

    R_j = max_b Q*(s_j, b) − Q*(s_j, a_j)     (>= 0)

Summed over the blind steps, R_1 + ... + R_{k-1} is the value lost by committing (performance-difference lemma, up to
discounting). So a gate with a fixed compute budget should stop where R_j is large. The decision oracles measure
exactly this: reg uses the critic at the real s_j in place of Q*, and act is the size of the action change.

## 2. Why the usual "hallucination" target is the wrong one

The return-error label is |imagined value − real value|. It can be large while R_j = 0 and small while R_j is large:

- **Value offset.** Shift every Q(z, ·) by a constant c. The return error grows by about c, but the ranking of actions,
  and therefore R_j, is unchanged. A return-error gate spends compute replanning where nothing would change.
- **Small error, flipped decision.** If two actions are nearly tied at s_j, a tiny state error can swap them. The
  return error is tiny, but R_j equals the true gap between them.

State-space disagreement (D) fails the same way. Heads can disagree along directions in which Q and pi are flat (no
decision change), and agree while all of them are wrong in the same direction (a shared error).

Empirically (docs/results_commitment_confirm.md): return-error oracle +0.054 over random, act +0.102, reg +0.145.

## 3. Imagined regret estimates R_j without seeing s_j

The robot does not know s_j, but its dynamics heads give plausible latents z^1_j .. z^M_j (head m imagines the plan on
its own). IR averages the critic's advantage gap of the stale action over them:

    IR_j = mean over m, h of max(0, Q_h(z^m_j, pi(z^m_j)) − Q_h(z^m_j, a_j))

**Proposition (informal, an upper-bound surrogate).** Let A(z, a) = max(0, Q(z, pi(z)) − Q(z, a)). Assume:
(i) the critic is eps-accurate at the real state: |Q(s, b) − Q*(s, b)| <= eps for all b;
(ii) pi is delta-greedy for Q: Q(s, pi(s)) >= max_b Q(s, b) − delta;
(iii) A is Lipschitz in z with constant Lip.
Then for every head m

    R_j <= A(z^m_j, a_j) + Lip * dist(s_j, z^m_j) + 2 eps + delta

and so R_j <= IR_j + Lip * (mean_m dist(s_j, z^m_j)) + 2 eps + delta.

Proof sketch: R_j = max_b Q*(s,b) − Q*(s,a) <= max_b Q(s,b) − Q(s,a) + 2 eps <= Q(s,pi(s)) − Q(s,a) + 2 eps + delta
<= A(s,a) + 2 eps + delta, then move from s to z^m with (iii) and average over m. (Q at a real state means Q at
its encoding, and dist is measured in latent space. IR takes max(0, .) per critic head before averaging, which is at
least A computed with the mean critic, so the bound holds for IR as defined.)

**What it says.**
- Where IR is small and the imagined states are close to reality, committing is provably cheap: the gate can trust
  those steps, whatever the return error is. A value offset (section 2) cancels inside A, so IR is immune to it.
- The slack terms are model error (dist(s, z)) and critic error (eps). These are exactly what our Bellman audit (LBA)
  detects, so detection and IR are complementary: the audit says when the bound is loose, IR says how much is at stake.
- Averaging over heads covers the true state better than one head, but the main term is the advantage gap itself.
  The single-head ablation IR1 (frontier study, G4) tests this.

## 4. Why gating on the running maximum is the right rule

The gate commits to the longest prefix whose running-max score stays below a threshold, with the threshold set by the
compute budget. If the scores were exact regrets, minimising total regret subject to a mean number of plans per
episode is a knapsack problem. The threshold rule is its greedy solution: replan where the marginal regret of
continuing exceeds the marginal cost of a planning call. That shadow price is the threshold, which the budget
controller sets.

## 5. Predictions, and how the experiments test them

| Prediction | Test | Status |
| :-- | :-- | :-- |
| Gating on regret beats gating on return error | reg/act vs oracle (K2) | confirmed |
| A label-free regret estimate captures most of the oracle's value | IR vs act (reported) | −0.012 [−0.040, +0.018] |
| IR beats state-space and Bellman-error signals | IR vs D, LBA (F3) | confirmed |
| The gain holds at every budget and beats fixed chunking at equal compute | G1, G2 (frontier) | running |
| Ranking and flow signals from the 2026 literature are weaker | G3 (frontier) | running |
| The ensemble is not essential | G4 (frontier) | running |
