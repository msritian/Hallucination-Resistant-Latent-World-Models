# Pre-registration (powered study): hallucination-aware value expansion

Written 2026-10-13 before any run of this study. Same code and settings as the pilot
(docs/preregistration_value_expansion.md, src/training/value_expansion.py): stock_aux TD-MPC2, 300k steps, expansion from
step 50k, torch.compile off, evaluation every 25k steps (10 episodes), final 100 episodes (seeds 1000-1099).
New seeds 80-84 (pilot seeds 70/71 are not reused). "fixed" (always trust) is dropped: the pilot showed it fails (0.22).

Motivation: STEVE-style trust from ensemble disagreement assumes the ensemble alarm grows with imagination error;
docs/results_compounding_final.md shows it does not (15/15 robots, 30/30 DMC tasks). Prediction: Bellman-audit trust
(lba) beats ensemble trust (D), and the gap is larger for longer expansion (h=10).

Runs (55): A StackCube h=5 {none, const, D, lba} x 5 seeds; B PickCube h=5 {none, const, D, lba} x 5 seeds;
C StackCube h=10 {const, D, lba} x 5 seeds.

Metrics per run: learning speed = mean evaluation success after 50k steps; final success (100 episodes).
Statistics: paired by seed, mean difference with a 95% bootstrap CI over seeds (pooled over A and B for P1-P3: 10 pairs).
- P1 (main): lba vs D, learning speed, A+B pooled: CI lower bound > 0.
- P2: lba vs none, learning speed, A+B pooled: CI lower bound > 0.
- P3: lba vs const, learning speed, A+B pooled: CI lower bound > 0.
- P4 (compounding link): in C, lba - D (speed) > 0 with CI lower bound > 0, and larger than the same gap in A.
- Final success for every comparison is reported as a secondary metric.
All arms, curves and failed criteria are reported regardless. A smoke test (PickCube lba h=10, StackCube D h=10, 8k steps)
must finish with final_eval.json before the full study is submitted.
