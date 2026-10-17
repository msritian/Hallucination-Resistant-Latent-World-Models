# Pre-registration: Bellman audit of released pretrained TD-MPC2 world models (DeepMind Control)

Written 2026-10-08 before any result. Models: the authors' released MT30 checkpoints (48M, then 317M parameters; 30
DeepMind Control tasks), loaded with TD-MPC2's own config/env/agent code. Per task: 20 real episodes with the agent's
planner (seeds 11000+); first 10 calibrate the audit (same recipe, features and trees as our ManiSkill results), last 10
test; 12-step windows starting every 5 steps; simulator-reward labels (imagined vs real return; P99/P90 of calibration
one-step errors). Baselines needing no extra models: critic-head spread (B), ELVIS-style (E), raw one-step residual (A),
raw anchored residual (Aa). Ensembles do not exist for released models. Code: `src/dmc_audit.py`.

Claims (MT30-48M, tasks with >= 20 test positives):
- D1: mean step-stratified AUROC of the audit > every baseline (all hallucinations), and > 0.5 on gradual and sudden.
- D2: the audit is best on at least 2/3 of the tasks.
- D3 (scale): on MT30-317M the audit keeps D1.
Reported regardless: every task, sudden/gradual, positive rates, returns.
