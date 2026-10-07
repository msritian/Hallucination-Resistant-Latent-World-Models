# Pre-registration (pilot): Horde-style per-signal Bellman audits for a frozen world model

Written 2026-10-07 before any result. Robots: StackCube s10, YCB s10 (frozen). Signals: all state-observation dims,
standardised; per-signal heads (next-signal predictor C, general value function G, gamma_c = 0.9) trained on the newest
3000 stored training episodes; evaluation on the robot's 50 saved episodes (25 calibration / 25 test). Code: `src/horde.py`.

Promising (scale up and develop) if, averaged over the 2 robots:
- H1 (reward-free): trees on signal residuals only reach reward-hallucination AUROC >= 0.70 (reward critic not used).
- H2 (per signal): mean per-signal AUROC >= 0.70.
- H3 (what is wrong): localisation top-1 >= 3x chance.
Also reported: the reward-critic audit on the same windows, and the combination (does Horde add to it?).
