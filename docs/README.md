# Project Reference Docs

Simple-language reference for the project **Temporal Bellman Consistency Auditing for Hallucination-Resistant Latent World Models**.

All math is written as plain text (inside code blocks) so it reads the same everywhere.

## Read in this order

| # | File | What it answers |
| :-: | :--- | :--- |
| 1 | [01_big_picture.md](01_big_picture.md) | What problem, what idea, what's new, is it publishable |
| 2 | [02_how_it_works.md](02_how_it_works.md) | End to end: training → planning → checker → trust → score, with worked examples |
| 3 | [03_math_reference.md](03_math_reference.md) | Every formula in one place, with notation |
| 4 | [04_elvis_and_prior_work.md](04_elvis_and_prior_work.md) | ELVIS vs ours in detail, other closest work, what's novel |
| 5 | [05_experiments_and_ablations.md](05_experiments_and_ablations.md) | Phases, go/no-go test, arms, ablations, metrics, possible outcomes |
| 6 | [06_faq.md](06_faq.md) | Short answers to the questions that caused confusion |

## Formal documents (repo root)

| File | Role |
| :--- | :--- |
| `../Research_WM.md` | The research case (paper-style: problem, method, novelty, applications) |
| `../execution_final.md` | The build and evaluation plan (what to implement, how, in what order), including hypotheses (Section 2) and limitations (Section 10) |
| [`literature_review.md`](literature_review.md) (in this folder) | Prior work and novelty assessment, with sources |

These `docs/` files explain the same content more simply. If anything here disagrees with `execution_final.md`, the execution plan wins.
