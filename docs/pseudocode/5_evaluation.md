# 5. Evaluation (`run_preflight.py::evaluate_model`, `src/preflight/metrics.py`)

## Protocol per trained robot

```python
episodes = record_real_episodes(agent, n=50, seeds=2000..2049)    # agent's own planner, eval mode
cal, test = episodes[:25], episodes[25:]
fit everything (bias, thresholds, trees) on cal                   # nothing from test is used for fitting
for every length-12 window of every test episode (start at each real step, real actions):
    roll = imagine(...); labels via the answer key; scores from every detector (ours and baselines)
```

## Metric: step-stratified AUROC

```python
def auroc(scores, labels):                     # P(score of a random positive > score of a random negative)
    pairs = [(sp, sn) for sp in scores[labels==1] for sn in scores[labels==0]]
    return mean(1 if sp > sn else 0.5 if sp == sn else 0 for sp, sn in pairs)

def stratified_auroc(scores, labels, step):    # compare only steps at the SAME imagined step k
    num = den = 0
    for k in unique(step):
        m = step == k
        n = count(labels[m]==1) * count(labels[m]==0)
        if n: num += auroc(scores[m], labels[m]) * n; den += n
    return num / den                           # a detector that only knows "later = worse" scores 0.5
```

## Uncertainty and comparisons

```python
def bootstrap_ci(test_episodes, n=1000):       # resample whole test episodes with replacement
    return P2.5 / P97.5 of stratified_auroc over resamples

def paired_difference(ours, rival):            # same resampled episodes for both detectors
    return bootstrap of auroc_ours - auroc_rival; significant if the 95% interval excludes 0

report stratified_auroc for: all positives, sudden-only positives, gradual-only positives
```

## Robots and confirmation

```python
robots = 22 (4 tasks; seeds 10, 20, 30, 40; grounded and standard critic) -> main tables
robots = 8 new (seed 50), every rule fixed in advance (docs/preregistration_seed50.md) -> confirmation, all criteria passed
```
