# Hallucination-Resistant Latent World Models

**Temporal Bellman Consistency Auditing**: using a world-model agent's own critic to detect when its imagined future stops being trustworthy, so it can plan further ahead without being fooled.

Built on [TD-MPC2](https://github.com/nicklashansen/tdmpc2) and [ManiSkill3](https://github.com/haosulab/ManiSkill) robot manipulation tasks.

## The problem

A model-based agent plans by *imagining* futures with a learned world model: from the current state and a candidate action sequence, it predicts the next latent states and rewards, step by step, and picks the plan with the highest predicted return.

Each imagined step is built on the previous one, so small errors compound. After a few steps the model can **hallucinate** physically impossible transitions, such as an object teleporting into the goal. A hallucinated plan promises large fake rewards, the planner selects it, and the robot fails. This is why planners like TD-MPC2 only look 3 steps ahead by default.

## The approach

### 1. Check each imagined step with the critic

The agent already has a critic `Q(z, a)` estimating future reward. On real experience it satisfies the Bellman equation, so for each imagined step `z_t → z_t+1` we measure the mismatch:

```
delta_t = | Q(z_t, a_t) − ( r_t + γ · Q(z_t+1, π(z_t+1)) ) |
```

Real physics changes value gradually and keeps this small. A hallucinated jump in value (e.g. "the cube is suddenly at the goal") breaks it.

### 2. Ground the critic in real experience

In standard TD-MPC2 the critic is partly trained on the world model's imagined states, so it can learn to accept the model's own mistakes. We train the critic (and policy) **only on latents encoded from real observations**. This keeps the critic's own inconsistency small, so the residual reflects the world model's errors.

### 3. Calibrate what "too large" means

Each imagined step `t` gets a threshold `τ(t)`, measured by replaying the world model along held-out real episodes and keeping only the replays that stayed accurate. `s_t = delta_t / τ(t)` then reads as "how many times larger than a correct prediction's error".

### 4. Score plans by trust

Each step gets a trust `w_t = exp(−max(0, s_t − 1))`; cumulative trust `Ω_t = w_0 · … · w_t` only drains. A plan's score counts imagined rewards only as far as they are trusted and, where trust is lost, uses the critic's estimate from the last believable state:

```
score = Σ_t γ^t [ Ω_t · r_t  +  (Ω_{t−1} − Ω_t) · Q(z_t, a_t) ]  +  γ^H · Ω_{H−1} · Q(z_H)
```

With full trust this is the standard planner score; with trust lost at step `k` it is exactly truncation at `k`. Blending imagined returns with a critic's estimate is a standard tool (the λ-return; used by STEVE and ELVIS). What is new here is **how trust is decided**.

## What is new vs prior work

| Approach | How it decides trust | Extra models |
| :--- | :--- | :--- |
| Ensembles (PETS, M2AC, STEVE) | Disagreement between several world models | Yes |
| ELVIS (2026) | Disagreement between several critics about a *state* | Critic ensemble |
| MOBILE (2023) | Spread of Bellman targets across a dynamics ensemble (offline RL) | Yes |
| **Ours** | Bellman consistency of each imagined *step*, from a grounded critic, calibrated on real replays | **None** |

A focused literature review is in `docs/literature_review.md`.

## Experiments

- **Tasks:** ManiSkill3 PushCube, PickSingleYCB, PegInsertionSide; PickCube and StackCube as alternatives. State observations, end-effector control at 20 Hz.
- **Models:** stock TD-MPC2 vs grounded TD-MPC2 (plus 4 auxiliary dynamics heads used only by the ensemble baseline), trained from scratch.
- **Phase 2, detection test ("preflight"):** replay the world model along real episodes, plant hallucinations (latent noise, teleport toward the goal, jumps to equal-value states), label the model's natural errors, and measure how well each signal detects them (AUROC). Signals compared: our residual, critic-ensemble spread, ELVIS's signal, dynamics-ensemble disagreement (PETS), MOBILE-style target spread.
- **Phase 3, planning test:** same trained models, planning horizons 3/6/12/24, scored by the standard sum, our trust score, ELVIS-style λ-return, or ensemble-based trust. Success rate over 50 fixed-seed episodes per arm.

## Results so far

**PushCube (pilot, seed 10):** both agents solve the task (100% success).

| Signal (grounded model) | Teleports | Natural errors (latent-error labels) | Natural errors (value-error labels) |
| :--- | :-: | :-: | :-: |
| **Ours (Bellman residual)** | **0.71** | 0.68 | **0.80** |
| Critic-ensemble spread | 0.56 | 0.65 | 0.73 |
| ELVIS-style signal | 0.40 | 0.34 | 0.33 |
| Dynamics ensemble (PETS) | 0.62 | **0.95** | 0.60 |

- Under the original pre-registered gate (≥ 0.80 on teleports and on latent-error labels), PushCube was a **NO-GO**.
- A follow-up diagnostic found that only **37%** of the model's large latent errors materially change value. On value-changing errors, the residual leads the no-extra-model signals, while the dynamics ensemble mostly flags errors that do not affect the planner.
- Grounding reduced the critic's own noise (median residual with the real next state: 0.044 vs 0.071 stock) and raised the residual's AUROC on value-changing errors (0.80 vs 0.64).
- For the remaining tasks the gate was **amended before their results** (2026-09-27): the primary criterion uses value-error labels, and the original gate is still reported.

**Harder tasks:** training in progress. Details and caveats are in `docs/05_experiments_and_ablations.md`.

## Engineering notes

- **Rendering on CHTC:** containers get CUDA but not the NVIDIA Vulkan driver, so SAPIEN renders with Mesa's CPU Vulkan (lavapipe, Mesa ≥ 24.3), selected by PCI address (`src/envs/lavapipe.py`).
- **Speed:** single-env PickSingleYCB and PegInsertionSide rebuild the scene at every reset by default (57% / 31% of training time). Training rebuilds every 10 episodes; evaluation keeps the default.
- **torch.compile:** with TD-MPC2's `reduce-overhead` mode, each recompilation left the update permanently slower. Runs use `--compile_mode default` for the update (`src/tools/update_bench.py`).
- **Preemption:** checkpoints (including the replay buffer) are mirrored to CHTC staging every 25k steps, and jobs resume from the newest copy.

## Repository layout

| Path | Contents |
| :--- | :--- |
| `docs/` | Plain-language guides to the method, math, prior work (incl. ELVIS), experiments, and a literature review |
| `src/auditor/` | Bellman residual and comparison signals, trust weights, ELVIS-style baseline, calibration, hallucination injection, source-fix loss |
| `src/agents/` | Grounded TD-MPC2 (critic/policy trained on real latents, optional extra dynamics heads) |
| `src/planning/` | Audited MPPI planner (separate planning horizon; standard / trust / ELVIS-style scoring) |
| `src/envs/` | ManiSkill3 wrapper for TD-MPC2 (state observations, 20 Hz, CPU Vulkan rendering) |
| `src/preflight/` | Phase 2 detection test and the value-error diagnostic |
| `src/train.py`, `src/evaluate.py` | Training with checkpoint/resume; evaluation of planner arms |
| `src/tools/` | Benchmarks (e.g. update speed around checkpoints and recompiles) |
| `tests/` | Unit and CPU end-to-end tests with mock models |
| `cluster/` | CHTC / HTCondor container recipe, job files, and run lists (see `cluster/README.md`) |
| `third_party/tdmpc2` | TD-MPC2, pinned as a git submodule |

## Running it

**Tests (CPU, no GPU needed):**

```bash
git clone --recurse-submodules <repo-url>
cd Hallucination-Resistant-Latent-World-Models
python3 -m venv .venv && .venv/bin/pip install torch numpy pytest tensordict
.venv/bin/python -m pytest -q tests
```

**Training and evaluation** need an NVIDIA GPU on Linux. On CHTC (HTCondor) the workflow is:

```bash
./cluster/pack_code.sh                                    # bundle src/ + TD-MPC2 for jobs
condor_submit train.sub queue_file=runs_phase1.txt        # train (one run per line)
condor_submit preflight.sub task=PushCube-v1 grounded=p1_grounded_push_s10 stock=p1_stock_push_s10
condor_submit eval.sub name=p1_grounded_push_s10 run=grounded task=PushCube-v1 preset=pilot
```

`cluster/README.md` covers building the container and the one-time setup. On other GPU machines, the same entry points run directly:

```bash
python -m src.train --run grounded --task PushCube-v1 --seed 10 --steps 200000 --out_dir runs/push_grounded
python -m src.preflight.run_preflight --task PushCube-v1 --grounded_model runs/push_grounded/final_model.pt \
    --stock_model runs/push_stock/final_model.pt --out preflight_push
```
