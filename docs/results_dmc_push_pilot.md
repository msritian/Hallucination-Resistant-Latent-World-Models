# DMC disturbance pilot: results

Protocol: docs/protocol_dmc_push_pilot.md (commit 2e256c7). Fixed chunking only, 20 episodes per arm, seeds
7100-7119, score = return / 1000. Results at strength 1.0 were 7 of 10 tasks when the rule was applied; the rule uses
the smaller strength first, so the missing 3 cannot change the decision.

```

push strength 0.5:
task                   k1     k4     k6 | k1-k4  k1-k6 [95% CI]
cartpole-balance    0.994  0.991  0.989 | +0.002 +0.005 [+0.004,+0.006] 
cheetah-run         0.124  0.122  0.109 | +0.001 +0.015 [-0.026,+0.055] 
cup-catch           0.974  0.968  0.958 | +0.006 +0.016 [+0.005,+0.029] 
finger-turn-hard    0.935  0.888  0.861 | +0.047 +0.074 [-0.076,+0.250] QUALIFIES
fish-swim           0.230  0.238  0.213 | -0.008 +0.018 [-0.019,+0.056] 
hopper-stand        0.493  0.416  0.393 | +0.076 +0.100 [+0.032,+0.165] QUALIFIES
pendulum-swingup    0.586  0.519  0.474 | +0.067 +0.111 [+0.072,+0.156] QUALIFIES
reacher-hard        0.981  0.972  0.952 | +0.010 +0.030 [+0.023,+0.037] 
walker-run          0.478  0.435  0.379 | +0.043 +0.099 [+0.081,+0.117] QUALIFIES
walker-walk         0.957  0.937  0.902 | +0.020 +0.055 [+0.034,+0.080] QUALIFIES
qualifying: 5 ['finger-turn-hard', 'hopper-stand', 'pendulum-swingup', 'walker-run', 'walker-walk']

push strength 1.0:
task                   k1     k4     k6 | k1-k4  k1-k6 [95% CI]
cartpole-balance    0.986  0.977  0.968 | +0.009 +0.018 [+0.013,+0.022] 
cheetah-run         0.132  0.132  0.081 | -0.000 +0.051 [+0.018,+0.088] QUALIFIES
cup-catch           0.973  0.967  0.955 | +0.005 +0.018 [+0.006,+0.034] 
hopper-stand        0.445  0.359  0.371 | +0.086 +0.074 [+0.020,+0.131] QUALIFIES
reacher-hard        0.975  0.927  0.908 | +0.048 +0.067 [+0.052,+0.083] QUALIFIES
walker-run          0.436  0.372  0.344 | +0.064 +0.092 [+0.068,+0.116] QUALIFIES
walker-walk         0.895  0.842  0.790 | +0.053 +0.104 [+0.080,+0.129] QUALIFIES
qualifying: 5 ['cheetah-run', 'hopper-stand', 'reacher-hard', 'walker-run', 'walker-walk']
```

**Decision (as pre-committed):** at push strength 0.5, 5 tasks qualify (at least 4 needed): finger-turn-hard,
hopper-stand, pendulum-swingup, walker-run, walker-walk. The gate study runs on these 5 tasks at strength 0.5
(docs/preregistration_dmc_push.md).
