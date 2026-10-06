# 3. Answer key: what counts as a hallucination (`run_preflight.py::return_error`, `real_label_analysis`)

Uses only rewards, which the model is trained to predict, plus the real rewards of the same actions. No probe, no
reconstruction, no critic.

```python
def imagination_error(roll, r_real):            # r_real[k] = real reward of action a_k from the same real state
    E = []
    for k in range(H - 1):                      # step k judges the transition into z_hat[k+1]
        E.append(abs(sum(gamma**(j-1) * (roll.r_hat[j] - r_real[j]) for j in 1..k+1)))
    return E                                     # r_hat[0] excluded: z[0] is the real encoded observation

def thresholds(E_calibration):                  # from the calibration half only
    e0 = [E[0] for E in E_calibration]          # one-transition errors = the model's normal accuracy
    return eps_hall = P99(e0), eps_clean = P90(e0)

def label(E_k, eps_hall, eps_clean):
    if E_k > eps_hall:   return 1               # hallucinated
    if E_k <= eps_clean: return 0               # correct
    return EXCLUDED                             # ambiguous; not used for training or scoring

def kind(E, k, big_jump):                       # big_jump = P99 of calibration increments E[j] - E[j-1]
    return "sudden" if any(E[j] - E[j-1] > big_jump for j in 0..k) else "gradual"
```

The simulator is deterministic given its state, so one real execution gives the real reward (a saved-state replay
reproduces a real step exactly in over 99% of steps). In a stochastic environment, `r_real` would be the average
over several replays.
