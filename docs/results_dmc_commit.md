# DMC commitment study: results

Pre-registration: docs/preregistration_dmc_commit.md (commit f8b3a1e; two simulator-snapshot bugs fixed after smoke
jobs, design unchanged). Released TD-MPC2 mt30-48M, 10 tasks, 15 arms, 30 episodes per arm. HTCondor cluster
8413098, all 10 jobs exit 0. Every claim is reported.

## Summary: all four claims fail, and the setting has almost no headroom

| Claim | Result | 95% CI | Verdict |
| :-- | :-: | :-: | :-: |
| D1 IR@4 − random@4 | −0.006 | [−0.018, +0.004] | fail |
| D2 act@4 − oracle@4 | +0.002 | [−0.009, +0.013] | fail |
| D3 IR@4 − ENT@4 | −0.006 | [−0.019, +0.005] | fail |
| D4 IR@4 − k4 | −0.017 | [−0.029, −0.007] | fail (IR is worse than fixed chunking) |

**Why.** The pre-registration flagged this case in advance: if committing does not hurt, no gate has anything to
recover. Here k1 − k4 = +0.011 [−0.001, +0.022], against +0.108 on ManiSkill. Even the simulator decision oracle
gains only +0.012 over random (+0.102 on ManiSkill), and every gate lies within 0.73-0.75. Five tasks are at
ceiling under every arm (cartpole-balance .99, reacher-hard .98, walker-walk .95-.98, cup-catch .94-.98,
pendulum-swingup .82-.84).

**Findings that stand anyway.**
- **A negative for IR:** in DMC, IR@4 is worse than fixed k4 (−0.017, CI below 0) and worse than the act oracle
  (−0.018). Variable commitment itself costs a little here (random@4 − k4 = −0.011), and IR does not recover it.
- PE, which sees the real state while acting, is the only gate that beats random (+0.018 [+0.008, +0.030]).
- IR is the most accurate before-acting score on decision labels (mean AUROC 0.652, vs ENT 0.548, B 0.547,
  FI 0.540, RS 0.490), but this accuracy does not turn into control gains when there is nothing to gain.
- On return-error labels every single-head score is near chance (0.45-0.54).

**Reading for the paper story.** The decision-regret result is established on ManiSkill manipulation, where blind
commitment is costly. It does not transfer to DMC locomotion with a released multitask model, where commitment is
nearly free. This bounds the claim. A fair test of transfer needs a setting where k1 − k_b is large, decided before
looking at gate results.

## Full report (`python -m src.tools.dmc_commit_report`)

## 1. Score per arm

| Arm | cartpole-balance | cheetah-run | cup-catch | finger-turn-hard | fish-swim | hopper-stand | pendulum-swingup | reacher-hard | walker-run | walker-walk | Mean | Actions/plan |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| k1 | 0.995 | 0.156 | 0.974 | 0.942 | 0.516 | 0.652 | 0.834 | 0.981 | 0.486 | 0.979 | **0.752** | 1.00 |
| k2 | 0.995 | 0.109 | 0.972 | 0.842 | 0.523 | 0.633 | 0.830 | 0.981 | 0.515 | 0.979 | **0.738** | 2.00 |
| k4 | 0.995 | 0.146 | 0.970 | 0.938 | 0.522 | 0.597 | 0.824 | 0.981 | 0.467 | 0.970 | **0.741** | 4.00 |
| random@2 | 0.995 | 0.137 | 0.969 | 0.910 | 0.530 | 0.584 | 0.836 | 0.981 | 0.495 | 0.974 | **0.741** | 1.98 |
| random@4 | 0.994 | 0.122 | 0.958 | 0.897 | 0.543 | 0.585 | 0.825 | 0.979 | 0.431 | 0.962 | **0.730** | 4.01 |
| IR@2 | 0.995 | 0.114 | 0.967 | 0.932 | 0.557 | 0.599 | 0.835 | 0.981 | 0.462 | 0.968 | **0.741** | 2.00 |
| IR@4 | 0.994 | 0.126 | 0.961 | 0.860 | 0.544 | 0.546 | 0.830 | 0.980 | 0.440 | 0.955 | **0.724** | 4.00 |
| oracle@4 | 0.995 | 0.141 | 0.944 | 0.931 | 0.526 | 0.616 | 0.820 | 0.981 | 0.461 | 0.972 | **0.739** | 3.99 |
| act@4 | 0.995 | 0.134 | 0.969 | 0.901 | 0.525 | 0.627 | 0.828 | 0.981 | 0.487 | 0.964 | **0.741** | 3.99 |
| reg@4 | 0.995 | 0.106 | 0.969 | 0.900 | 0.583 | 0.547 | 0.823 | 0.981 | 0.490 | 0.972 | **0.737** | 3.99 |
| ENT@4 | 0.994 | 0.116 | 0.972 | 0.928 | 0.532 | 0.577 | 0.832 | 0.977 | 0.423 | 0.947 | **0.730** | 3.99 |
| B@4 | 0.994 | 0.124 | 0.974 | 0.891 | 0.534 | 0.594 | 0.830 | 0.979 | 0.427 | 0.957 | **0.730** | 3.99 |
| FI@4 | 0.995 | 0.115 | 0.962 | 0.903 | 0.549 | 0.584 | 0.819 | 0.977 | 0.423 | 0.939 | **0.727** | 3.99 |
| RS@4 | 0.994 | 0.117 | 0.961 | 0.866 | 0.552 | 0.567 | 0.824 | 0.976 | 0.419 | 0.950 | **0.723** | 4.00 |
| PE@4 | 0.995 | 0.130 | 0.976 | 0.939 | 0.530 | 0.652 | 0.830 | 0.981 | 0.471 | 0.976 | **0.748** | 3.99 |

## 2. Pre-registered claims

- **D1** IR@4 − random@4 = -0.006 [-0.018, +0.004] (10 tasks) → failed
- **D2** act@4 − oracle@4 = +0.002 [-0.009, +0.013] (10 tasks) → failed
- **D3** IR@4 − ENT@4 = -0.006 [-0.019, +0.005] (10 tasks) → failed
- **D4** IR@4 − k4 = -0.017 [-0.029, -0.007] (10 tasks) → failed

## 3. Reported, not claimed

- k1 − k4: +0.011 [-0.001, +0.022] (10 tasks)
- k1 − k2: +0.014 [+0.001, +0.028] (10 tasks)
- act@4 − random@4: +0.012 [+0.003, +0.021] (10 tasks)
- reg@4 − random@4: +0.007 [-0.006, +0.019] (10 tasks)
- oracle@4 − random@4: +0.009 [-0.002, +0.021] (10 tasks)
- ENT@4 − random@4: +0.000 [-0.010, +0.011] (10 tasks)
- B@4 − random@4: +0.001 [-0.012, +0.013] (10 tasks)
- FI@4 − random@4: -0.003 [-0.015, +0.009] (10 tasks)
- RS@4 − random@4: -0.007 [-0.020, +0.006] (10 tasks)
- PE@4 − random@4: +0.018 [+0.008, +0.030] (10 tasks)
- IR@4 − act@4: -0.018 [-0.030, -0.007] (10 tasks)
- IR@4 − reg@4: -0.013 [-0.024, -0.003] (10 tasks)
- IR@4 − B@4: -0.007 [-0.021, +0.007] (10 tasks)
- IR@4 − FI@4: -0.003 [-0.017, +0.010] (10 tasks)
- IR@4 − RS@4: +0.001 [-0.014, +0.016] (10 tasks)
- IR@4 − PE@4: -0.024 [-0.038, -0.013] (10 tasks)
- IR@2 − random@2: -0.000 [-0.009, +0.010] (10 tasks)
- IR@2 − k2: +0.003 [-0.009, +0.017] (10 tasks)
- IR@2 − k1: -0.011 [-0.020, -0.001] (10 tasks)
- random@4 − k4: -0.011 [-0.023, -0.000] (10 tasks)

## 4a. Accuracy on return labels (calibration plans, AUROC)

| Task | IR | ENT | B | FI | RS | PE |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| cartpole-balance | 0.226 | 0.551 | 0.586 | 0.619 | 0.456 | 0.591 |
| cheetah-run | 0.371 | 0.498 | 0.355 | 0.683 | 0.518 | 0.616 |
| cup-catch | 0.421 | 0.434 | 0.551 | 0.500 | 0.483 | 0.561 |
| finger-turn-hard | 0.626 | 0.631 | 0.590 | 0.626 | 0.478 | 0.728 |
| fish-swim | 0.361 | 0.273 | 0.363 | 0.470 | 0.518 | 0.488 |
| hopper-stand | 0.552 | 0.729 | 0.646 | 0.629 | 0.411 | 0.535 |
| pendulum-swingup | 0.621 | 0.073 | 0.058 | 0.322 | 0.228 | 0.077 |
| reacher-hard | 0.600 | 0.302 | 0.579 | 0.344 | 0.490 | 0.455 |
| walker-run | 0.509 | 0.484 | 0.570 | 0.494 | 0.497 | 0.662 |
| walker-walk | 0.572 | 0.442 | 0.598 | 0.392 | 0.466 | 0.654 |
| **Mean** | **0.486** | **0.442** | **0.490** | **0.508** | **0.454** | **0.537** |

## 4b. Accuracy on decision labels (calibration plans, AUROC)

| Task | IR | ENT | B | FI | RS | PE |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| cartpole-balance | 0.537 | 0.428 | 0.491 | 0.603 | 0.533 | 0.426 |
| cheetah-run | 0.660 | 0.431 | 0.550 | 0.412 | 0.484 | 0.556 |
| cup-catch | 0.697 | 0.549 | 0.534 | 0.538 | 0.498 | 0.512 |
| finger-turn-hard | 0.468 | 0.591 | 0.579 | 0.604 | 0.506 | 0.596 |
| fish-swim | 0.449 | 0.431 | 0.526 | 0.591 | 0.506 | 0.520 |
| hopper-stand | 0.788 | 0.803 | 0.777 | 0.756 | 0.514 | 0.833 |
| pendulum-swingup | 0.831 | 0.476 | 0.438 | 0.474 | 0.442 | 0.472 |
| reacher-hard | 0.805 | 0.732 | 0.471 | 0.528 | 0.493 | 0.522 |
| walker-run | 0.602 | 0.488 | 0.543 | 0.446 | 0.459 | 0.638 |
| walker-walk | 0.678 | 0.551 | 0.560 | 0.450 | 0.461 | 0.535 |
| **Mean** | **0.652** | **0.548** | **0.547** | **0.540** | **0.490** | **0.561** |
