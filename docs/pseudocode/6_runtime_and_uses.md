# 6. Run time, and how we tried to use the detector

## At run time (`src/planning/gpu_audit.py`: the same trees, re-implemented exactly on the GPU)

```python
# inside MPPI, for any set of candidate plans (512 per iteration)
p = detect(trees, o_t, candidate_actions)       # p[k, n] for every step k and candidate n
```

## Uses tested (all pre-registered; same robots and seeds for every arm)

```python
# (a) Warning light (src/monitor.py): on the plan about to be executed
plan = mppi_H12(o_t)
if max_k p(plan)[k] > threshold(rate):  act with mppi_H3(o_t)  else: act with plan
# result: accurate on executed plans (AUROC 0.81) but success equals a random warning at the same rate

# (b) Inside the search (src/planning/detector_planner.py)
score_n = lambda_return(r_hat, v, lam=0.8 * (1 - p[:, n]))                 # audit-weighted trust
drop the 25% of candidates with the largest max_k p[k, n] before choosing elites
# result: no gain vs constant lambda; dropping hurt (-0.044), worse than random dropping

# (c) Training (src/training/audit_replay.py): every 50k steps
score every stored 12-step window by max_k p; keep the top 30%
each batch = 50% slices from flagged windows + 50% uniform
# interim (StackCube): no gain; slightly higher imagination error

# (d) Idea 1, running (src/rerank.py): correct plan scores instead of avoiding plans
top16 = best 16 candidates of MPPI
over_estimate_n = regression_trees(plan_features(top16[n]))  # learned from real outcomes: imagined - true return
execute argmax_n (imagined_score_n - over_estimate_n)
# compared with: imagined score, ensemble penalty (lambda tuned), oracle (true score), random, standard H3
```
