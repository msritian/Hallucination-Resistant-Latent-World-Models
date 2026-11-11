# Pre-registration: decision regret on a second domain, with a released checkpoint (DMC commitment study)

Written on 2026-11-11, before any result of this experiment existed. The commit that adds this file is the timestamp.
Every result is reported, pass or fail.

## Why

Every commitment result so far is on ManiSkill manipulation with our own grounded TD-MPC2 robots. This study asks
whether the findings hold (a) on a different domain (DeepMind Control locomotion and control), (b) for a model we did
not train: the released TD-MPC2 multitask checkpoint mt30-48M from TD-MPC2's authors, and (c) with a single dynamics
head (released checkpoints have no ensemble), so only single-head signals are possible.

## Design (code: `src/commit_dmc.py`, job: `cluster/dmc_commit.sh`)

- **Tasks (10):** walker-walk, walker-run, cheetah-run, hopper-stand, reacher-hard, cartpole-balance, cup-catch,
  finger-turn-hard, fish-swim, pendulum-swingup (the 10 tasks of the earlier DMC compounding study). One job per task.
- **Planner:** TD-MPC2's MPPI (eval mode) with a 12-step horizon, as in the ManiSkill studies; after each planning call
  the agent executes k plan actions blind, then replans, with the warm start shifted by k.
- **Score:** episode return / 1000 (500 agent steps, action repeat 2).
- **Seeds:** evaluation 6000–6029 (30 episodes per arm, identical for every arm: the task's random state and the
  torch seed are reset per episode); calibration 6900–6904 (5 closed-loop k = 1 episodes, every plan also executed in
  a simulator copy).
- **Gates:** same commitment rule (longest prefix whose running-max score stays below a threshold), same per-level
  thresholds and budget controller as the ManiSkill studies. Scores (single head, defined in the code's docstring):
  IR (imagined regret), ENT (policy log-std at the imagined state: the action-entropy signal of adaptive action
  chunking for VLA policies), B (critic-head disagreement), FI (latent-flow instability), RS (rank instability among
  the 16 best MPPI candidates), PE (prediction error while acting); oracles from the simulator copy: oracle (return
  error), act, reg.
- **Arms (15 per task):** k1, k2, k4, random@2, random@4, IR@2, IR@4, oracle@4, act@4, reg@4, ENT@4, B@4, FI@4, RS@4,
  PE@4.

## Claims (decided now)

- **D1. IR helps on a released model in a new domain:** score(IR@4) − score(random@4) > 0, 95% CI above 0.
- **D2. Decision-relevant errors are the costly ones here too:** score(act@4) − score(oracle@4) > 0, 95% CI above 0.
- **D3. IR beats the action-entropy signal:** score(IR@4) − score(ENT@4) > 0, 95% CI above 0.
- **D4. IR beats fixed chunking at equal compute:** score(IR@4) − score(k4) > 0, 95% CI above 0.

**Statistics.** As in the ManiSkill studies, with tasks in place of robots: differences paired by seed within each
task, averaged within tasks and then over tasks; 95% CIs from a bootstrap that resamples episodes within each task.

**Reported, not claimed:** every arm's mean score per task; act, reg, oracle, B, FI, RS, PE against random@4 and
IR@4; IR@2 against random@2 and k2; k1 − k4 (does commitment hurt in DMC at all?); each score's step-stratified AUROC
on the calibration plans for return-error labels (top quarter vs bottom half of the return error, since the released
model has no fitted audit) and decision labels (act top quarter vs bottom half); achieved actions per plan.

If k1 − k4 is not positive (committing does not hurt), D1–D4 are still reported, but there is little for any gate to
recover in this domain.

## Procedure

A smoke job (walker-walk, 1 episode per arm, 1 calibration episode) checks the pipeline. Its numbers are not used.
Amendments are allowed only before the full jobs start, only for problems the smoke job finds, and are committed with
a reason.
