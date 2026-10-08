# Pre-registration (pilot): hallucination-aware value expansion for TD-MPC2

Written 2026-10-08 before any result. Standard TD-MPC2 (with 4 extra dynamics heads that never shape the main model, so
the ensemble arm exists), trained online on StackCube for 300k steps, seeds 70 and 71, torch.compile off for every arm.
From step 50k, the critic's target for each real transition mixes 0..5-step imagined returns (policy rollouts from the
real next latent), cutting imagination at the first untrusted step in expectation (`src/training/value_expansion.py`).
Arms: none (standard target), fixed (always 5 steps), const (trust 0.5 everywhere), D (ensemble disagreement, STEVE /
MACURA-style), lba (our Bellman audit, refitted every 50k steps on the newest 50 episodes). D and lba use the same
per-step empirical-CDF mapping to trust, so their average trust is equal and only WHERE they trust differs.

Measures: evaluation success every 25k steps (10 episodes) and final success (100 episodes, seeds 1000-1099).
Promising (scale to more tasks/seeds) if, averaged over the 2 seeds:
- V1: lba's mean success over evaluations after 50k steps (learning speed) > const and > D; and
- V2: lba's final success >= none.
All arms and curves are reported regardless.
