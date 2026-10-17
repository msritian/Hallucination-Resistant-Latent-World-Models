# 2. Signals

All signals are computed from the imagined rollout only (no simulator, no real future). Index `k = 0..H-1`
(the learned audit uses `k = 0..L-1`).

## Bellman consistency: the core check

```python
# value of this step  should equal  imagined reward + discounted value of the next imagined state
Delta[k] = r_hat[k] + gamma * v[k+1] - q[k]          # signed residual (src: Rollout.signed_residual)
```

## Our 14 signals (`src/preflight/run_preflight.py::plan_signals`, `src/auditor/signals.py`)

```python
A[k]   = abs(Delta[k])                                                     # one-step residual: sudden errors

for n in (2, 3, 5, 8):                                                     # n-step residuals (A2, A3, A5, A8)
    s = max(0, k - n + 1)
    A_n[k] = abs(q[s] - (sum(gamma**(j-s) * r_hat[j] for j in s..k) + gamma**(k-s+1) * v[k+1]))

Aa[k]  = abs(q[0] - (sum(gamma**j * r_hat[j] for j in 0..k) + gamma**(k+1) * v[k+1]))   # anchored at the REAL start z[0]: gradual drift
Ac[k]  = abs(sum(gamma**j * Delta[j] for j in 0..k))                       # cumulative signed: noise cancels, drift adds up
Ao[k]  = max(0, Delta[k])                                                  # optimistic part only
P[k]   = std_i(r_hat[k] + gamma * v_head[k+1][i] - q_head[k][i])           # residual spread across critic heads
At[k]  = abs(q[k] - (r_hat[k] + gamma * expected_min_of_2_target_heads(z[k+1])))  # vs the critic's own training target
B[k]   = std_i(v_head[k+1][i])                                             # critic disagreement (also a baseline)
E[k]   = v[k+1] + beta * std_i(v_head[k+1][i])                             # ELVIS-style UCB, beta = 1 (also a baseline)

# bias-corrected versions; bias[k] = median signed residual on calibration steps that are verifiably clean
Ab[k]  = abs(Delta[k] - bias[k])
Acb[k] = abs(sum(gamma**j * (Delta[j] - bias[j]) for j in 0..k))
```

```python
def critic_bias(agent, calibration_episodes):                    # src: calibration_bias
    windows = all length-H windows of the calibration episodes (real start, real actions, real next observations)
    roll    = imagine(...) for each window
    latent_err[k] = || z_hat[k+1] - encode(o_real[k+1]) ||
    clean[k]      = latent_err[j] <= P90(latent_err[0]) for all j <= k     # clean prefix
    return [median(Delta[k] over windows with clean[k]) for k in range(H)]
```

## Baselines (computed on the same windows)

```python
# need 4 extra dynamics heads d_1..d_4 (trained alongside; never shape the encoder)
D[k] = mean_over_latent_dims( std_m( d_m(z[k], a_k) ) )                    # dynamics-ensemble disagreement (PETS-style)
M[k] = std_m( r_hat[k] + gamma * V(d_m(z[k], a_k)) )                      # MOBILE-style Bellman-target spread
B[k], E[k]                                                                # as above (no extra models)
```
