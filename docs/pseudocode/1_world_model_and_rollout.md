# 1. World model and imagined rollout

## TD-MPC2 parts used (trained, frozen)

```python
z      = encode(o)                 # real observation -> latent (no decoder exists)
z_next = dynamics(z, a)            # deterministic latent dynamics
r_hat  = reward(z, a)              # predicted reward (two-hot over 101 bins -> scalar)
q_i    = Q_i(z, a), i = 1..5       # critic ensemble (two-hot -> scalar)
a_pi   = pi_mean(z)                # policy prior, mean action
V(z)   = mean_i Q_i(z, pi_mean(z)) # state value used everywhere below
```

How TD-MPC2 trains its critic (for reference; `third_party/tdmpc2/tdmpc2/tdmpc2.py::_update`):

```python
# a real slice from the replay buffer: o_0..o_3, a_0..a_2, r_0..r_2
z_0 = encode(o_0)                                   # real
z_hat_1 = dynamics(z_0, a_0); z_hat_2 = dynamics(z_hat_1, a_1); z_hat_3 = dynamics(z_hat_2, a_2)   # imagined
targets_t = r_t + gamma * min_of_2_random_target_heads(encode(o_{t+1}), pi(encode(o_{t+1})))       # REAL r, REAL o
loss_Q = sum_t rho^t * CE(Q_i([z_0, z_hat_1, z_hat_2][t], a_t), targets_t)   # inputs partly imagined, targets real
# (our "grounded" variant feeds encode(o_t) instead of z_hat_t, with no gradient into encoder/dynamics)
```

## Imagined rollout (`src/auditor/signals.py::audit_rollout`)

```python
def imagine(o_t, actions):                    # actions: a_0..a_{H-1}
    z = [encode(o_t)]                         # z[0] is REAL (encoded observation)
    for k in range(H):
        z.append(dynamics(z[k], actions[k]))  # z[1..H] are IMAGINED
    r_hat  = [reward(z[k], actions[k])            for k in range(H)]
    q_head = [[Q_i(z[k], actions[k]) for i in 1..5] for k in range(H)]
    q      = [mean(q_head[k])                       for k in range(H)]
    v_head = [[Q_i(z[k], pi_mean(z[k])) for i in 1..5] for k in range(H + 1)]
    v      = [mean(v_head[k])                       for k in range(H + 1)]
    return Rollout(z, r_hat, q_head, q, v_head, v)
```

During detection the actions are the ones the robot really executed (so the real outcome is known). At run time
they can be any candidate plan.
