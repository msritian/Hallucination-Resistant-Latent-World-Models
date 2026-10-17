# Pre-registration (pilot): ensemble-free pessimism for offline latent MPC

Written 2026-10-08 before any result. Robots: TD-MPC2 (with 4 extra dynamics heads, never shaping the main model) trained
OFFLINE for 200k updates on medium-replay-style datasets: StackCube (first 5000 episodes of an online run), YCB (first
10000). Planning: MPPI, H=3. Arms: no penalty; penalties LBA (ours, fitted only on the dataset), D (MOReL/PETS-style),
M (MOBILE-style), B (critic-head spread, edge-of-reach style), each at lambda 0.5/1/2 (scale-free, per MPPI iteration).
50 evaluation episodes per arm, seeds 10000+. Code: `src/offline_plan.py`.

Motivation: the edge-of-reach analysis (Sims et al. 2024) attributes offline model-based failure to value errors at
imagined states the critic never learned; the Bellman audit measures the critic's inconsistency at imagined states
directly, from a single model.

Go (scale to more tasks/datasets, add training-time pessimism) if, averaged over the 2 robots, the best LBA arm beats
no penalty by >= 0.10 success AND is >= the best arm of each ensemble penalty (D, M, B). Lambda is reported per arm; the
headline uses lambda = 1 for every penalty (fixed in advance) to avoid picking the best lambda after the fact.
