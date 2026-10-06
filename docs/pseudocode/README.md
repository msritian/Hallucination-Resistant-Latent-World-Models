# Pseudocode: the Bellman-consistency hallucination detector

Python-style pseudocode of what the code does, step by step. Each block names the real function it describes.
Notation: `H` = imagined horizon (12), `L = H - 1` = audited steps, `K = 5` critic heads, `gamma = 0.95`.

| File | Contents |
| :-- | :-- |
| `1_world_model_and_rollout.md` | the TD-MPC2 parts used, and how an imagined rollout is computed |
| `2_signals.md` | the 14 Bellman-consistency signals (our method) and the 4 baselines |
| `3_answer_key.md` | how a hallucination is labelled with the simulator (real rewards) |
| `4_learned_audit.md` | the 31 features and the gradient-boosted trees: fit once, then predict |
| `5_evaluation.md` | per-robot protocol, step-stratified AUROC, bootstrap, sudden vs gradual |
| `6_runtime_and_uses.md` | using the detector at run time, and the planner/training experiments we ran |

End-to-end, for one trained robot:

```python
agent    = load_trained_tdmpc2(robot)                      # frozen; the detector never changes it
episodes = record_real_episodes(agent, n=50)               # observations, actions, real rewards
cal, test = episodes[:25], episodes[25:]

bias              = critic_bias(agent, cal)                 # 2_signals.md
X_cal, rollouts   = audit_features(agent, cal, bias)        # 2_signals.md + 4_learned_audit.md
y_cal, eps        = answer_key(rollouts, cal)               # 3_answer_key.md
trees             = fit_trees(X_cal, y_cal)                 # 4_learned_audit.md

X_test, rollouts  = audit_features(agent, test, bias)
y_test, _         = answer_key(rollouts, test, eps)         # thresholds from the calibration half
p_test            = trees.predict_proba(X_test)             # hallucination probability per imagined step
score             = stratified_auroc(p_test, y_test)        # 5_evaluation.md  (ours: ~0.82)
```
