# 6. Run time

At run time the detector needs only the trained world model and the trees fitted in `4_learned_audit.md`. No simulator,
no real future and no rewards are used.

```python
def detect(trees, bias, o_t, actions):          # actions: any candidate sequence (12 steps)
    roll = imagine(o_t, actions)                 # 1_world_model_and_rollout.md
    sig  = signals(roll, bias)                   # 2_signals.md
    return [trees.predict_proba(features(roll, sig, k)) for k in range(L)]   # p_k = P(imagined step k is hallucinated)
```

For many candidates at once (e.g. all 512 MPPI candidates per planning iteration), `src/planning/gpu_audit.py` evaluates
the same trees exactly on the GPU. Every job checks the GPU copy against sklearn before use (max difference 0.0).

Uses of the detector beyond detection (planning, training, shift monitoring, policy evaluation, per-signal audits) were
tested separately. Their designs and results are in `docs/preregistration_*.md` and the experiment log, not in this
pseudocode.
