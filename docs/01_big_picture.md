# 01 — The Big Picture

## 1. The problem in one paragraph

A robot with a **world model** can "imagine" the future: given the current state and a plan of actions, the world model predicts the next states and rewards, step by step. The robot's **planner** imagines many plans, scores them, and picks the best.

The trouble: each imagined step is built on the previous imagined step, so small errors pile up. After a few steps the world model can **hallucinate** — for example, "the cube teleports into the goal". A hallucinated plan promises huge fake rewards, so the planner picks it, and the robot fails in reality.

That's why TD-MPC2 (the base system we use) only imagines **3 steps** ahead by default, even though many manipulation tasks need longer lookahead.

```
imagined:  z0 → z1 → z2 → z3 → z4 ...
                 small error → bigger error → hallucination (impossible physics)
```

## 2. The core idea in one sentence

> The robot's own critic, if it is trained only on real experience, can spot when an imagined step breaks physics — because a hallucinated step violates the Bellman equation. We use that check to trust imagination only as far as it is believable, which should let the robot plan much further ahead.

## 3. The three ingredients

| # | Ingredient | Simple meaning |
| :-: | :--- | :--- |
| 1 | **Bellman residual checker** | For each imagined step, ask: "does the value change from this state to the next make sense?" A big mismatch = likely hallucination. |
| 2 | **Grounded critic** | Train the critic only on real states and real rewards, so it has never "learned" the world model's mistakes. |
| 3 | **Trust-weighted score** | Count imagined rewards only as much as they are trusted; where trust is lost, use the critic's realistic estimate instead. |

## 4. Where the project is used ("vectors")

| Vector | What | Status |
| :--- | :--- | :--- |
| **A — Live control** | ManiSkill3 robot tasks (PushCube, PickSingleYCB, PegInsertionSide). The checker runs inside the planner. | **Main result** |
| **B — Video evaluation** | Check videos of robot policies for "teleport" glitches that make bad policies look good. Simulation-only demo. | Secondary |
| **Source fix** | Train the world model itself to hallucinate less, using the residual as an extra loss. | Secondary (prior work exists) |

## 5. What is new vs borrowed

### Borrowed (we cite it)

| Piece | Prior work |
| :--- | :--- |
| Planning by imagining many plans (MPPI) | TD-MPC2 and earlier |
| Blending imagination and critic by a trust dial (the scoring formula) | λ-return (Sutton 1988), STEVE (2018), **ELVIS (2026)** |
| "Only trust the model where it is reliable" | STEVE, M2AC, MACURA, ELVIS |
| Critic trained on real data used in a planner | LOOP (2021) |
| Training the world model to be value-consistent (source fix) | VAML, value equivalence, VIPO |

### New (as far as our literature review found)

| # | Piece | Why it matters |
| :-: | :--- | :--- |
| **1** | **The critic's own Bellman residual, on each imagined step, as the trust signal in planning** | Others decide trust with extra models (ensembles), a separate error-predictor, image reconstruction, or critic disagreement. Ours needs nothing extra. |
| **2** | **Grounding is what makes the signal valid** | A testable claim: the checker works with a real-data critic and works worse with a normal one. |
| **3** | **Per-step thresholds calibrated against real model errors** | "Normal error" is measured on replays where we know the truth, not hand-tuned. |
| 4 | Vector B: Bellman-based correction of video-model policy rankings | Problem is known; this correction is untried. Demo only. |

## 6. Is it publishable?

**Novel ≠ publishable.** The signal is new, but ELVIS (May 2026) already blends imagination and critic inside the planner. The only real difference is **how trust is decided**. So the paper stands only if **our signal beats ELVIS's signal**.

| If the experiments show… | Then… |
| :--- | :--- |
| Our signal clearly beats ELVIS's and planning improves at long horizons | Solid paper |
| Our signal ties ELVIS's but is cheaper than ensembles | Weak / workshop paper |
| Our signal loses | Not a method paper; maybe an analysis paper |

That's why a cheap **go/no-go test (Phase 2)** runs before the expensive training. See [05_experiments_and_ablations.md](05_experiments_and_ablations.md).

## 7. Known limitations (say these openly)

1. **Blind spot:** a fake jump between two states of *equal* value doesn't break the Bellman equation, so the checker misses it.
2. **The critic defines "consistent":** a wrong but self-consistent critic can hide errors. Grounding reduces this, doesn't remove it.
3. **Test-time only:** all planners are compared on models trained with the normal short-horizon planner.
4. **Vector B is a demo** with simulated glitches, not proof about real video models.
5. **Simulation only**, no hardware.
6. **Shared components:** the scoring formula (with ELVIS) and the source fix (with VAML/VIPO) are not our contribution.
