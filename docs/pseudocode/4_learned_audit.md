# 4. Learned audit: features and trees (`run_preflight.py::_features`, `learned_audits`, `fit_lba`)

## Features: 31 numbers per imagined step k

```python
SIGNALS = [A, Aa, Ac, Ab, Acb, P, B, E, At, Ao, A2, A3, A5, A8]            # 14 signals (2_signals.md)

def features(roll, signals, k):
    x = []
    for s in SIGNALS:
        x += [s[k], max(s[0..k])]                # value now, and the highest so far in this rollout
    x += [roll.v[k+1], roll.r_hat[k], k]         # imagined value, predicted reward, step index
    return x                                     # 14*2 + 3 = 31
```

## Fit once per robot (calibration half), then freeze

```python
def fit_trees(calibration_windows):
    X, y = [], []
    for w in calibration_windows:                # every length-12 window of the 25 calibration episodes
        roll = imagine(w.o_start, w.actions)
        sig  = signals(roll, bias)
        E    = imagination_error(roll, w.r_real)
        for k in range(L):
            lab = label(E[k], eps_hall, eps_clean)
            if lab is not EXCLUDED:
                X.append(features(roll, sig, k)); y.append(lab)
    return GradientBoostedTrees(n_trees=200, max_leaves=15, learning_rate=0.05, l2=1.0,
                                class_weight="balanced", seed=0).fit(X, y)    # settings fixed for every robot
```

## Predict (no labels, no simulator)

```python
def detect(trees, o_t, actions):
    roll = imagine(o_t, actions)
    sig  = signals(roll, bias)
    return [trees.predict_proba(features(roll, sig, k)) for k in range(L)]   # p_k = P(step k hallucinated)
```

Variants reported: the same features with logistic regression ("linear"); a single raw signal such as `Aa`; and
"no Bellman" trees using only `v`, `r_hat` and `k` (0.77 vs 0.82, showing that the residuals carry the result).
