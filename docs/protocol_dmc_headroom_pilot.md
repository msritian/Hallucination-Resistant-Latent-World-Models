# DMC headroom pilot: protocol (decided before any pilot data)

Written on 2026-11-12, after the DMC commitment study (docs/results_dmc_commit.md: no headroom, k1 − k4 = +0.011)
and before any pilot data existed. The commit that adds this file is the timestamp.

**Purpose.** Find a DMC setting where blind commitment is costly, using only fixed-k arms (no gate is run), so that a
fair gate study can then be pre-registered there.

**Pilot.** Released mt30-48M, the 5 tasks not at ceiling in the DMC study: cheetah-run, finger-turn-hard, fish-swim,
hopper-stand, walker-run. Fixed k = 1, 4, 6, 8. 30 episodes each on NEW seeds 7000-7029 (not the evaluation seeds of
any study). Same planner and code (`src/commit_dmc.py +fixed_ks=1+4+6+8 +seed0=7000`).

**Decision rule (fixed now).**
- A task qualifies if its mean score k1 − k6 >= 0.05 (return/1000).
- If at least 3 tasks qualify, a gate study is pre-registered on the qualifying tasks at budget 6, with fresh evaluation
  seeds (8000 onward), the same gates as the DMC study, and claims D1-D4 restated at budget 6.
- If fewer than 3 qualify, DMC is reported as a domain where commitment is nearly free, and no gate study is run.
The pilot never runs a gate, so it cannot select a setting that favours IR.
