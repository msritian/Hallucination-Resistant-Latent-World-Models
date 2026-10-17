# Pre-registration (Phase A): relevance-aware detection of test-time distribution shift

Written 2026-10-07 before any result. Robots: StackCube s10 and s30 (grounded, trained nominally). The robot acts with
TD-MPC2's own 3-step planner, 20 episodes per condition (seeds 8000+): none, gain:0.5, gain:0.25, bias:0.3, mass:3,
mass:10, obs:0.5, obs:0.1. The audit is fitted on the robot's NOMINAL saved episodes only. Code: `src/shift.py`.

A condition is HARMFUL if its success is at least 0.20 below nominal, HARMLESS if within 0.05 (decided from the data
of this run). Signals: LBA (ours), D, M (ensembles), B (critic spread), PE1/PE (observed latent prediction error,
needs the future observation, i.e. an after-the-fact monitor).

Go to Phase B (detector-triggered adaptation / fallback) if, averaged over the 2 robots:
- S1: on harmful shifts, LBA window AUROC (shifted vs nominal) >= 0.75 and >= every ensemble baseline; and
- S2 (relevance): on harmless shifts LBA AUROC is closer to 0.5 than PE's, i.e. it alarms less on changes that do not
  hurt the task; and
- S3: across conditions, mean LBA score correlates with the success drop more strongly than PE (Spearman).
If PE matches LBA on S2 and S3, the audit adds nothing over a plain prediction-error monitor here, and we report it.
