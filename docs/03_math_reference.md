# 03 — Math Reference

All formulas in plain text. `×` is multiplication, `|x|` is absolute value, `exp` is the exponential function.

## Notation

| Symbol | Meaning |
| :--- | :--- |
| `z_t` | Latent state at step t (512 numbers). Real at t = 0; imagined after that |
| `a_t` | Action at step t |
| `r_t` | Reward predicted by the world model for step t |
| `d(z, a)` | World model (dynamics): predicts the next latent |
| `Q(z, a)` | Critic: expected future reward from z taking a. Always the **average of the 5 heads** |
| `policy action` | The policy's average (mean) action at a state |
| `gamma` | Discount = 0.95 for our tasks (TD-MPC2 sets it from episode length) |
| `H` | Planning horizon (default 12 for us; TD-MPC2 default is 3) |
| `tau(t)` | Normal residual at step t, calibrated on real replays |
| `w_t` | Trust in step t alone (0 to 1) |
| `Omega_t` | Cumulative trust up to and including step t |
| `kappa` | Strictness knob for trust (default 1) |

## 1. Critic value (TD-MPC2 decoding)

```
for each head k:
   probs_k   = softmax( head_k([z, a]) )                  101 buckets
   x_k       = sum over buckets ( probs_k × bucket_value )
   Q_k       = symexp(x_k) = sign(x_k) × (e^|x_k| − 1)
Q(z, a)      = average of Q_1 .. Q_5
```

## 2. Grounded critic training

```
target  = r_t(real) + gamma × Q_target( z_t+1(real), policy action )
loss    = cross-entropy between Q(z_t(real), a_t) and target   (TD-MPC2 uses bucket regression)

z_t(real) = encoder(real observation), with encoder gradient blocked for this loss
```

## 3. Bellman residual (the checker)

```
delta_t = | Q(z_t, a_t) − ( r_t + gamma × Q(z_t+1, policy action at z_t+1) ) |
```

## 4. Normalized suspicion

```
s_t = delta_t / tau(t)
```

## 5. Calibration

```
e_t        = || z_t(imagined along real actions) − encoder(real s_t) ||      true latent error
eps_clean  = 90th percentile of one-step errors e_1
eps_hall   = 99th percentile of one-step errors e_1
clean at t = e_j ≤ eps_clean for all j ≤ t+1
tau(t)     = max( 95th percentile of delta_t on clean replays , 0.000001 )
if fewer than 200 clean samples at step t:  tau(t) = tau(t−1)
```

## 6. Trust

```
w_t        = exp( −kappa × max(0, s_t − 1) )
Omega_t    = w_0 × w_1 × ... × w_t
Omega_−1   = 1
```

## 7. Trust-weighted score (our planner score)

```
Score = sum_{t=0}^{H−1}  gamma^t × [ Omega_t × r_t + (Omega_{t−1} − Omega_t) × Q(z_t, a_t) ]
        + gamma^H × Omega_{H−1} × Q(z_H, policy action)
```

Same score written backwards (shows it is a λ-return with λ_t = w_t). Forward and backward give the **same number** — it's only a different order of adding:

```
G_H = Q(z_H, policy action)                                   value after the plan ends
G_t = (1 − w_t) × Q(z_t, a_t)  +  w_t × ( r_t + gamma × G_{t+1} )   score from step t to the end
Score = G_0
```

Check (Research_WM example, gamma = 1, rewards 1,1,1,10,10,10, end 10, trust 0.5 at step 3, Q(z3,a3) = 4):
`G6 = 10, G5 = 20, G4 = 30, G3 = 0.5×4 + 0.5×(10+30) = 22, G2 = 23, G1 = 24, G0 = 25` — same 25 as the forward form.

Diagnostic: effective horizon `H_eff = Omega_0 + Omega_1 + ... + Omega_{H−1}`.

## 8. ELVIS score (for comparison, Arm 4b)

```
V_k(z)   = Q_k(z, policy action)                     per head
mu_t     = average of V_k(z_t),   sigma_t = std of V_k(z_t)
UCB_t    = mu_t + beta × sigma_t
norm_t   = (clip((UCB_t − running_mean)/running_std, −3, 3) + 3) / 6       in [0, 1]
lambda_t = lambda_max − (lambda_max − lambda_min) × norm_t

G_H = mu_H
G_t = r_t + gamma × ( (1 − lambda_t) × mu_{t+1}  +  lambda_t × G_{t+1} )
Score = G_0
```

Ours vs ELVIS when a step is fully distrusted:

```
ours  (w_t = 0):       G_t = Q(z_t, a_t)                    from BEFORE the step
ELVIS (lambda_t = 0):  G_t = r_t + gamma × mu_{t+1}          still uses the imagined reward + next state

difference = | Q(z_t, a_t) − ( r_t + gamma × mu_{t+1} ) | = delta_t     (the Bellman residual)
```

## 9. Other detection signals (compared in Phase 2)

```
A (ours):   delta_t
B:          std over critic heads of Q(z_t+1, policy action)
C (ours+):  max( s_A , s_B )             each normalized by its own tau(t)
D (PETS):   average over latent dims of std over 5 dynamics heads of d_k(z_t, a_t)
E (ELVIS):  UCB_{t+1}
M (MOBILE): std over 5 dynamics heads of [ r_t + gamma × Q(d_k(z_t, a_t), policy action) ]
```

## 10. Source fix loss (secondary)

```
L_aud  = [ sum_{t=0}^{11} gamma^t × delta_t ] / [ sum gamma^t ] / critic_value_scale
aux    = coef × L_aud
cap    = 0.1 × (consistency_coef × consistency_loss)          (no gradient through cap)
aux    = aux × min(1, cap / aux)                                (rescale, keep gradient direction)
total  = TD-MPC2 loss + ramp(step) × aux
ramp   = 0 for first 25% of training, linear to 1 over next 10%
```

Only the dynamics network gets gradient from this loss.

## 11. Vector B scoring

```
delta_video_t = | Q_obs(z_t, a_t) − ( r_obs(z_t, a_t) + gamma × Q_obs(z_t+1, a_t+1) ) |
                  (z = DINOv2 frame features; a_t+1 = the video's real next action)

delta_db_t    = max(0, delta_video_t − policy's average delta at step t on clean videos)
tau_video(t)  = 95th percentile of delta_db_t on clean videos
w, Omega      = as in section 6 (kappa = 1)

V_audited(policy) = average over videos of  Success_video × Omega_last
```
