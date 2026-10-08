# Pre-registration (pilot): consistency scores as trust indicators for off-policy evaluation

Written 2026-10-08 before any result. Six trained robots (StackCube s10/s20/s30, YCB s10/s20/s30; grounded, with
dynamics heads). Ten latent-space target policies of varied quality and shift (pi; noise 0.15/0.3/0.6/1.0/2.0; xyz
offset 0.15/0.3/-0.3; random). Per policy, 30 start states (seeds 9000+): the off-policy estimate is the mean imagined
return in the world model; the truth is the mean real return of exact real twins (same start, same noise). Trust scores
per imagined episode (mean over steps): target-policy (SARSA-style) Bellman residual (sarsa), behaviour-policy residual
(A), anchored residual on the first chunk (Aa), dynamics-ensemble disagreement (D), critic spread (B); and the trained
audit's max (LBA). Code: `src/imag_eval.py --ope`, `src/tools/ope_trust_report.py`.

Claim O1: averaged over the 6 robots, the policy-level Spearman between the target-policy residual and the absolute
estimation error is >= 0.5 and higher than that of D and B (ensemble / critic spread).
Reported regardless: every score, policy and episode level, per-policy errors.
