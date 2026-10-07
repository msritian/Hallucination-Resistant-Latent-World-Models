# Pre-registration (pilot): when to trust imagined policy evaluation

Written 2026-10-07 before any result. Robots: StackCube s10 and YCB s10 (grounded; D/M available). Six latent-space
policies of varied quality (pi, pi + noise 0.3/0.6/1.0, pi + xyz bias 0.3, random), 30 start states each (seeds 9000+),
every imagined episode paired with its real twin (same start, same noise). Code: `src/imag_eval.py`; analysis:
`src/tools/imag_eval_report.py`.

Go (scale to more robots/tasks, then a full pre-registration) if, averaged over the 2 robots:
- E1 (trust): LBA's area under the risk-coverage curve is lower than that of D, M and B, and lower than random (mean gap).
- E2 (benefit): estimating each policy from its LBA-trusted half gives a lower mean absolute error than using all episodes
  and than the D- and M-trusted halves, with Spearman over policies not lower than with all episodes.
All numbers are reported regardless.
