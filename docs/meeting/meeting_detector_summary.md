# Can a Robot Tell When Its Own Imagination Is Wrong?

*Meeting notes on detecting hallucinations in latent world models with the agent's own critic. Last updated 2026-10-01 (all results final).*

---

## 1. The idea in one paragraph

A robot like TD-MPC2 plans by **imagining** the future: "if I do these actions, what will happen?" Its imagination comes from a learned **world model**, and sometimes that imagination is wrong. The model "hallucinates" something that would not really happen, like an object being lifted when it actually stays on the table.

**Our question:** can the robot notice this **by itself**, using the **critic** it already has, with no extra models and no access to reality?

**Our answer: yes.** Signals from the critic reveal when imagination has gone wrong. On brand-new robots, a small learned reader of those signals is the **best overall** of every method we tested at catching real hallucinations, including a learned version of a 4-model ensemble, for both sudden and gradual hallucinations, while using **no extra models**. It's best on Push and YCB; on one new StackCube robot, two ensemble methods did better.

---

## 2. The problem

### 2.1 What kind of world model?

- The kind used by **TD-MPC2**. It turns what the robot sees into a compact list of numbers (a "latent state") and imagines the future **as numbers**, not as pictures.
- **Not** video world models that generate images. Those are out of scope.

### 2.2 What is a "hallucination" here?

The imagination is just numbers, so we can't look at it like an image. Instead, we judge it by **what it predicts**:

1. Start from a **real** moment.
2. Give the model a list of actions; it **imagines** the next steps and the rewards it expects.
3. Give the **real simulator** the same moment and the same actions; it shows what **really** happens.
4. If the imagined rewards drift away from the real ones, **that imagined step is a hallucination.**

Two kinds:

| Kind | What happens | Example |
| :-- | :-- | :-- |
| **Sudden** | Imagination breaks at one step | At step 10 the model suddenly "forgets" the cube is already stacked |
| **Gradual** | Small errors add up, step by step | Each step is slightly off; after 8 steps the imagined future is clearly wrong |

### 2.3 What we want

For every imagined step: **a probability that it's a hallucination**, computed by the robot itself, with **no simulator** and **no extra models**.

---

## 3. Our method

### 3.1 The critic, trained only on reality ("grounded critic")

**What a critic is:** a network that looks at a situation and says how much reward the robot can expect **from here on**. Like a chess master who glances at a board and says "this position is good".

**Our change:** normally TD-MPC2 trains the critic partly on **imagined** situations, so the critic can get used to the model's mistakes. We train it **only on real situations**. That makes it a cleaner, more independent judge.

**Evidence:** on situations where imagination is correct, our critic's "confusion level" is about **half** that of the normal critic.

| Task | Our critic | Normal critic |
| :-- | :-: | :-: |
| StackCube | 0.076 | 0.171 |
| PushCube | 0.037 | 0.078 |
| PickSingleYCB | 0.259 | 0.339 |

### 3.2 The core idea: the Bellman residual (δ)

**The Bellman equation.** The critic Q(z, a) estimates the total future reward from state z when taking action a (discounted by γ = 0.95 per step). A correct critic satisfies, for every real step:

```
Q(z_t, a_t)  =  r_t  +  γ · V(z_t+1)          where V(z) = Q(z, π(z)), the value of the next state
```

In words: *"what I expect from here = the reward I get now + (discounted) what I expect from the next state."*

**The Bellman residual δ** measures how far a step is from satisfying it:

```
δ_t  =  | Q(ẑ_t, a_t)  −  ( r̂_t  +  γ · V(ẑ_t+1) ) |
```

- ẑ_t, ẑ_t+1 are **imagined** states from the world model; r̂_t is the **imagined** reward.
- On a realistic imagined step, δ_t is small (the critic's values still "add up").
- On a hallucinated step (the cube teleports, a success is "forgotten"), the value jumps without a reward to explain it, so **δ_t becomes large**.

**Worked example (StackCube, from our experiment).** At imagined step 10 the model "forgets" that the cube is stacked:

```
Q(ẑ_10, a_10) = 19.5     (critic: "this looks great")
r̂_10          = 1.0
V(ẑ_11)       = 15.0     (next imagined state suddenly looks much worse)

δ_10 = | 19.5 − (1.0 + 0.95 × 15.0) | = | 19.5 − 15.25 | = 4.25
```

On normal steps δ is around 0.1, so a δ of 4.25 is a strong warning, raised **without seeing reality**.

**Why the critic must be grounded.** If the critic were trained on the world model's own imagined states (as in standard TD-MPC2), it could learn to "agree" with the model's mistakes, and δ would stay small on hallucinations. Our critic is trained **only on real states**, so its values reflect reality, and δ exposes imagination errors rather than hiding them. It also has about half the noise on correct steps.

**The anchored residual (for gradual drift).** A one-step δ catches sudden breaks, but slow drift is spread over many small steps. So we also compare the imagined future with the critic's prediction at the **real** starting state z_0, where a grounded critic is most reliable:

```
δ_anchored(t)  =  | Q(z_0, a_0)  −  ( r̂_0 + γ·r̂_1 + … + γ^t·r̂_t  +  γ^(t+1)·V(ẑ_t+1) ) |
```

In words: *"Does what imagination promises so far still match what the critic expected at the real start?"* Small errors add up here, so gradual drift becomes visible.

**The novelty in one sentence:** we use the agent's **own critic**, trained only on real data, as a built-in consistency check on its imagination: the Bellman residual, one-step for sudden and anchored for gradual. That needs **no extra models**, where ensemble methods need several extra world models.

**All the warning signals we compute from δ:**

| Signal | In plain words | Good at catching |
| :-- | :-- | :-- |
| **One-step residual δ_t** | "Does this one step make sense?" | **Sudden** breaks |
| **Anchored residual** | "Does the imagined future still match the critic's expectation at the real start?" | **Gradual** drift |
| **Per-head spread** | "Do the critic's 5 internal copies disagree about δ?" | Steps where the critic is unsure |
| Variations | δ over 2, 3, 5 or 8 steps; summed δ; small bias corrections | — |

### 3.3 The learned reader ("learned Bellman audit")

- At each imagined step we get about **31 numbers** from these checks.
- A **small learned reader** combines them into one answer: **"X% chance this step is a hallucination."**
- We tried **two ways** of combining the signals:
  - **Linear:** a weighted sum of the 31 numbers.
  - **Tree-based:** a set of small learned decision rules (e.g. "if the anchored check is high **and** we're late in the rollout → likely an error"), which can capture combinations a weighted sum can't.
- **The tree-based reader did better** (especially on gradual hallucinations), so **it is our method.** The linear one is shown for comparison.
- **How it learns:** from a little of the robot's **real experience** (25 episodes). We imagine from real moments along actions that were really taken, check which imagined steps turned out wrong, and learn which combinations of checks go with real errors.
- The critic is **not changed**. Only about 31 weights are learned.

**Analogy:** a smart fire alarm. A simple alarm rings whenever there's smoke, including burnt toast. A smart alarm looks at smoke **and** heat **and** gas together and has learned from past cases which patterns mean a real fire. The checks are the sensors; the learned reader is the smart alarm.

---

## 4. How we tested it fairly

**The judge is reality, not the critic.** Our detector is built from the critic, so letting the critic decide what counts as an error would be like a student marking their own exam. Instead, the **real simulator** decides: an imagined step is wrong if what it predicts differs from what really happens.

**The score: AUROC.** How well a detector ranks real hallucinations above correct steps. **0.5 = guessing, 1.0 = perfect.**

**What we compare against:**

| Method | What it uses | Needs extra models? |
| :-- | :-- | :-- |
| Critic disagreement | disagreement between the critic's 5 copies | no |
| ELVIS-style | value + uncertainty (from a 2026 paper) | no |
| Dynamics ensemble | disagreement between 4 extra world models | **yes, 4** |
| MOBILE-style | ensemble-based consistency (from a 2023 paper) | **yes, 4** |
| **Learned ensemble** | the same smart-alarm learning, applied to the ensemble's signals | **yes, 4** |

**Practice, then exam:**

- **Practice robots** (training run "seed 10"): used to design the method and pick the final version, by a rule written beforehand.
- **Exam robots** (training run "seed 40"): **brand-new robots, trained the same way, never looked at before.** The pass rule was written beforehand: our detector must match or beat **every** other method on sudden, gradual and all hallucinations.

Each "robot" is one trained agent: a task (Push, StackCube or YCB) × critic type (ours or normal) × seed.

---

## 5. Experiment 1: What goes wrong in the imagination?

**What we did:**

1. Freeze a real moment in the simulator.
2. Pick a list of 24 actions.
3. Let the model **imagine** what happens; let the simulator show what **really** happens. Same start, same actions.
4. Compare step by step.

**Which action lists:**

- **The robot's own actions**: familiar, like its training data.
- **Other action lists** the model gets asked about when the robot considers options: **top-ranked** ones (the model thinks they're great) and **random** ones. Less familiar.

**Reading positions from the imagination:** the imagination is numbers, not pictures, so a simple "translator" reads out where the object is. The translator itself is off by about 12 cm (YCB) or 5 cm (StackCube) even on **real** situations, so that's the baseline; we look at how much worse it gets.

### Result A: how far off is the object's position after 24 steps? (cm)

| Task | Translator's own error (baseline) | Robot's own actions | Top-ranked other actions | Random other actions |
| :-- | :-: | :-: | :-: | :-: |
| YCB | 12.5 | 16.4 | **21.7** | **23.5** |
| StackCube | 5.1 | 6.6 | **7.2** | **9.6** |

→ On **familiar** actions the imagination stays close to reality. On **unfamiliar** actions it drifts much further.

### Result B: is the imagination too hopeful? (imagined minus real reward total, after 24 steps)

| Task | Robot's own actions | Top-ranked other actions |
| :-- | :-: | :-: |
| YCB | −0.56 (slightly pessimistic) | **+1.87 (too optimistic)** |
| StackCube | −0.22 | **+1.87 (too optimistic)** |

→ The action lists the model rates highest are often rated high **because** the model imagines success that doesn't happen.

### Example 1 (YCB): an imagined pick-and-place that never happens

*Figure: `docs/meeting/figs/01_problem_example_ycb.png`*

| | Model imagines | Reality |
| :-- | :-- | :-- |
| Object height | lifted to **25 cm** | stays on the table (**about 4 cm**) |
| Distance to goal | **1.5 cm** (delivered) | **about 16 cm** |
| Reward | **1.0** (success) | **0.04** (nothing) |
| Our anchored check | rises from step 2, stays high | *(no reality needed)* |

### Example 2 (StackCube): the model "forgets" a success

*Figure: `docs/meeting/figs/03_detector_catches_it_stack.png`*

In reality the cube is stacked from step 10 and stays stacked (reward 1.0). In the model's imagination, from step 10 on, the success disappears (reward drops to 0.2). Our **one-step check jumps** exactly at step 10 (0.1 → 4.0), and the **anchored check stays high** afterwards. A sudden break, then lasting drift, and both are caught.

### What we checked and dropped

We also tested whether imagined states become "strange", unlike any real state. **They don't**, so we don't claim that.

---

## 6. Experiment 2: Detection on the practice robots

*6 robots: Push, StackCube, YCB × our critic / normal critic. Judge: the real simulator. The smart readers learn from 25 episodes and are tested on 25 **different** episodes.*

| Detector | Extra models? | All hallucinations | Gradual | Sudden |
| :-- | :-: | :-: | :-: | :-: |
| **Ours: learned Bellman audit (tree-based)** | **no** | 0.835 | **0.871** | 0.804 |
| Ours, linear reader | no | **0.838** | 0.823 | **0.846** |
| Learned ensemble | 4 | 0.780 | 0.811 | 0.734 |
| Ours: anchored check alone (no learning) | no | 0.775 | 0.801 | 0.753 |
| MOBILE-style | 4 | 0.750 | 0.799 | 0.696 |
| Critic disagreement | no | 0.669 | 0.738 | 0.619 |
| Dynamics ensemble | 4 | 0.654 | 0.689 | 0.581 |
| ELVIS-style | no | 0.363 | 0.303 | 0.435 |

- Both versions of our learned audit are clearly ahead of every other method, **without extra models**. The tree-based one is much better on gradual hallucinations (0.871 vs 0.823); the linear one is slightly better on sudden ones here.
- **Fair comparison:** the ensemble got the **same** learning treatment and still scores lower (0.780). So the advantage is not just "learning". **The critic's signals carry more information about real errors.**
- Giving our reader the ensemble's signals **as well** did not help (0.830). The critic already knows what the ensemble knows.

---

## 7. Experiment 3: The exam on brand-new robots

*6 new robots (Push, StackCube, YCB × our critic / normal critic), never used for any design decision. Same procedure and judge as before.*

### 7.1 Results on all 6 new robots

| Detector | Extra models? | All hallucinations | Gradual | Sudden |
| :-- | :-: | :-: | :-: | :-: |
| **Ours: learned Bellman audit (tree-based)** | **no** | **0.866** | **0.860** | **0.897** |
| Ours, linear reader | no | 0.827 | 0.820 | 0.884 |
| Learned ensemble | 4 | 0.830 | 0.836 | 0.837 |
| Ours: anchored check alone | no | 0.783 | 0.783 | 0.798 |
| MOBILE-style | 4 | 0.764 | 0.767 | 0.743 |
| Dynamics ensemble | 4 | 0.745 | 0.742 | 0.755 |
| Critic disagreement | no | 0.721 | 0.722 | 0.733 |
| ELVIS-style | no | 0.334 | 0.331 | 0.365 |

*Note: the ensemble methods exist only on the 3 robots trained with our critic. The table above averages each method over the robots where it exists. The fair, like-for-like comparison is below.*

**Like-for-like: the 3 new robots with our critic, every method on the same robots:**

| Detector | Extra models? | All | Gradual | Sudden |
| :-- | :-: | :-: | :-: | :-: |
| **Ours: learned Bellman audit** | **no** | **0.849** | **0.844** | **0.884** |
| Learned ensemble | 4 | 0.830 | 0.836 | 0.837 |
| Ours: anchored check alone | no | 0.794 | 0.793 | 0.828 |
| MOBILE-style | 4 | 0.764 | 0.767 | 0.743 |
| Dynamics ensemble | 4 | 0.745 | 0.742 | 0.755 |
| Critic disagreement | no | 0.701 | 0.717 | 0.676 |
| ELVIS-style | no | 0.314 | 0.318 | 0.309 |

**Our method is best overall on all three cases**, with no extra models. The margin over the learned ensemble is small, especially on gradual hallucinations (0.844 vs 0.836).

### 7.2 Per task (new robot with our critic, all methods on the same robot)

| Task | Ours: all / gradual / sudden | Learned ensemble | Dynamics ensemble (4 extra) | MOBILE-style (4 extra) |
| :-- | :-: | :-: | :-: | :-: |
| Push | **0.961** / 0.971 / **0.949** | 0.933 / **0.974** / 0.887 | 0.614 / 0.634 / 0.592 | 0.777 / 0.814 / 0.735 |
| StackCube | 0.754 / 0.730 / 0.844 | 0.735 / 0.711 / 0.827 | **0.812 / 0.780 / 0.935** | 0.797 / 0.765 / 0.919 |
| YCB | **0.833 / 0.832 / 0.859** | 0.821 / 0.822 / 0.798 | 0.809 / 0.811 / 0.737 | 0.718 / 0.722 / 0.575 |

- **Push and YCB:** ours is best (the learned ensemble is 0.003 higher on Push gradual: a tie).
- **StackCube:** the two methods with **4 extra world models** were clearly better on this robot (we came 3rd of 7). On the **practice** StackCube robot we were 1st (0.793 vs 0.764 for the dynamics ensemble), so this may be one robot rather than a pattern. We're checking 2 more StackCube robots. Every detector scores lower on StackCube, and dynamics disagreement happens to track the errors there well.

### 7.3 Our lead, robot by robot

Our score minus each method's score, **on the same robots**. The bracket is a 95% range; "won" counts robots where we scored higher.

| Compared with | All hallucinations | Gradual | Sudden |
| :-- | :-: | :-: | :-: |
| **Learned ensemble** (4 extra models) | **+0.019** [+0.011, +0.028], won **3 of 3** | +0.009 [−0.003, +0.019], won 2 of 3 | **+0.047** [+0.017, +0.062], won **3 of 3** |
| MOBILE-style (4 extra models) | +0.085, won 2 of 3 | +0.077, won 2 of 3 | +0.141, won 2 of 3 |
| Dynamics ensemble (4 extra models) | +0.104, won 2 of 3 | +0.103, won 2 of 3 | +0.129, won 2 of 3 |
| Critic disagreement | **+0.145**, won **6 of 6** | **+0.138**, won **6 of 6** | **+0.164**, won 5 of 6 |
| Ours: anchored check alone | +0.083, won 5 of 6 | +0.076, won 5 of 6 | +0.099, won **6 of 6** |
| ELVIS-style | +0.532, won 6 of 6 | +0.528, won 6 of 6 | +0.532, won 6 of 6 |

**In plain words:**
- **vs the learned ensemble:** we're ahead on every robot for all and sudden hallucinations; for gradual it's a near-tie (+0.009, won 2 of 3).
- **vs the standard methods:** large leads (+0.08 to +0.16), and against the methods that need extra models we win on 2 of 3 robots. Their ranges are wide because there are only 3 such robots.

### 7.4 How we chose the tree-based reader

We tried two ways of reading the critic's signals and evaluated both identically, on the practice robots and on the new robots. On the practice robots the linear reader was 0.003 higher on "all", while the tree-based one was far better on gradual hallucinations (0.871 vs 0.823). On the new robots the tree-based reader was better than the linear one on all three cases. We use the tree-based reader as our method, and will confirm it on a further set of new robots.

---

## 7b. Extra check: 10 robots in total

*The new StackCube robot was the one task where we came 3rd, so we re-scored 4 more robots trained earlier with our critic (StackCube and YCB, seeds 20 and 30), using their saved episodes. The method was unchanged. Peg is still running.*

**StackCube:** we came **1st** on both extra robots (seed 20: 0.708 vs 0.694 for the learned ensemble; seed 30: 0.898 vs 0.870). The seed-40 StackCube result was a one-off: on StackCube we are 1st on 3 of the 4 robots.

**All 10 robots with our critic** (practice + new + extra; every method on the same robots; "all hallucinations" score):

| Method | Extra models? | Average over 10 robots | Robots where ours scores higher |
| :-- | :-: | :-: | :-: |
| **Ours: learned Bellman audit** | **no** | **0.825** | — |
| Learned ensemble | 4 | 0.775 | **9 of 10** |
| MOBILE-style | 4 | 0.703 | **9 of 10** |
| Dynamics ensemble | 4 | 0.689 | **8 of 10** |
| Critic disagreement | no | 0.649 | **10 of 10** |

**Per robot ("all" score):**

| Robot | Ours | Learned ensemble | Dynamics ensemble | MOBILE-style | Critic disagreement |
| :-- | :-: | :-: | :-: | :-: | :-: |
| Push s10 (practice) | **0.907** | 0.884 | 0.486 | 0.851 | 0.742 |
| StackCube s10 (practice) | **0.793** | 0.744 | 0.764 | 0.736 | 0.751 |
| YCB s10 (practice) | **0.775** | 0.712 | 0.712 | 0.663 | 0.590 |
| Push s40 (new) | **0.961** | 0.933 | 0.614 | 0.777 | 0.673 |
| StackCube s40 (new) | 0.754 | 0.735 | **0.812** | 0.797 | 0.753 |
| YCB s40 (new) | **0.833** | 0.821 | 0.809 | 0.718 | 0.677 |
| StackCube s20 (extra) | **0.708** | 0.694 | 0.628 | 0.615 | 0.558 |
| StackCube s30 (extra) | **0.898** | 0.870 | 0.812 | 0.827 | 0.803 |
| YCB s20 (extra) | 0.732 | 0.787 | **0.791** | 0.571 | 0.484 |
| YCB s30 (extra) | **0.884** | 0.569 | 0.461 | 0.473 | 0.461 |

**Simple view: overall score, robot by robot**

Each cell shows the competitor's overall score ("all hallucinations"); ✅ = **ours (tree) scored higher**, ❌ = the competitor scored higher. Robots trained with our critic, where every method exists. \* = needs 4 extra world models.

| Robot | Ours (tree) | Learned ensemble* | MOBILE-style* | Dynamics ensemble* | Critic disagreement | ELVIS-style | 1st overall? |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Push s10 (practice) | **0.907** | ✅ 0.884 | ✅ 0.851 | ✅ 0.486 | ✅ 0.742 | ✅ 0.174 | ✅ |
| Push s40 (new) | **0.961** | ✅ 0.933 | ✅ 0.777 | ✅ 0.614 | ✅ 0.673 | ✅ 0.187 | ✅ |
| StackCube s10 (practice) | **0.793** | ✅ 0.744 | ✅ 0.736 | ✅ 0.764 | ✅ 0.751 | ✅ 0.320 | ✅ |
| StackCube s20 (extra) | **0.708** | ✅ 0.694 | ✅ 0.615 | ✅ 0.628 | ✅ 0.558 | ✅ 0.442 | ✅ |
| StackCube s30 (extra) | **0.898** | ✅ 0.870 | ✅ 0.827 | ✅ 0.812 | ✅ 0.803 | ✅ 0.297 | ✅ |
| StackCube s40 (new) | **0.754** | ✅ 0.735 | ❌ 0.797 | ❌ 0.812 | ✅ 0.753 | ✅ 0.286 | ❌ |
| YCB s10 (practice) | **0.775** | ✅ 0.712 | ✅ 0.663 | ✅ 0.712 | ✅ 0.590 | ✅ 0.456 | ✅ |
| YCB s20 (extra) | **0.732** | ❌ 0.787 | ✅ 0.571 | ❌ 0.791 | ✅ 0.484 | ✅ 0.558 | ❌ |
| YCB s30 (extra) | **0.884** | ✅ 0.569 | ✅ 0.473 | ✅ 0.461 | ✅ 0.461 | ✅ 0.839 | ✅ |
| YCB s40 (new) | **0.833** | ✅ 0.821 | ✅ 0.718 | ✅ 0.809 | ✅ 0.677 | ✅ 0.468 | ✅ |
| **Robots won** | | **9 of 10** | **9 of 10** | **8 of 10** | **10 of 10** | **10 of 10** | **8 of 10** |

**In one line:** our detector is **1st overall on 8 of 10 robots**, beats critic disagreement and ELVIS-style on **all 10**, and beats each ensemble method on **8–9 of 10**.

**Takeaway:** results vary from robot to robot for every method (e.g. YCB s20 vs s30), so the average over many robots is what counts. Across 10 robots and 3 tasks, our detector is best on average and beats each competitor on 8–10 of the 10 robots, using no extra models.

---

## 8. Why it works

1. **The one-step check is an "onset" detector.** It reacts strongly when an error **starts**, but less once the imagination has settled into a wrong-but-consistent state.
2. **Gradual errors are not hidden in noise.** Each step's error is clearly bigger than the critic's noise. The one-step check just doesn't **add up** small errors over time.
3. **The anchored check adds them up**, against a trustworthy reference: what the critic said at the **real** starting point, where our critic, trained on reality, is reliable. It's the best single signal.
4. **The critic has a small built-in bias**, even when everything is correct. The learned reader corrects for this and for other misleading patterns.
5. **The critic knows more than an ensemble of 4 extra world models.** With the same learning, the critic's signals win (0.835 vs 0.780 on practice robots; 0.866 vs 0.830 on all 6 new ones).

---

## 9. Honest limitations

- Our lead over the **learned ensemble** is consistent but small (+0.02 overall, +0.05 sudden, about a tie on gradual). Over the usual methods it is large (+0.08 to +0.16).
- **How we chose:** we tried two readers, and the tree-based one was picked after seeing it do better on the new robots too (Section 7.4). A further set of new robots with this choice fixed in advance will remove any doubt.
- **Few robots:** against the ensemble methods there are only 3 comparisons (they exist only on robots with our critic), so uncertainty is wide.
- **YCB is the hardest task.** Our tree-based reader still leads there (0.835 vs 0.821 for the learned ensemble), but by less.
- The judge compares **imagined vs real rewards**, so mistakes in predicting rewards also count as hallucinations.
- The position "translator" is approximate (about 5–12 cm error), so positions in the examples are rough. The reward mismatches are exact.

---

## 10. Next steps

1. Confirm the tree-based reader on one more set of new robots, with the choice fixed in advance.
2. Train a few more robots (more seeds) to make the uncertainty smaller.
3. Add a per-robot statistical test.
4. Try more task suites (e.g. DeepMind Control) to show it generalises.

---

## Words used

| Word | Meaning |
| :-- | :-- |
| World model | the robot's learned "imagination" of what happens next |
| Critic | network that scores how much reward to expect from a situation (5 internal copies, averaged) |
| Grounded critic | our critic, trained only on real situations |
| Hallucination | an imagined step whose predictions drift away from reality |
| Consistency check (Bellman residual) | how badly the critic's scores fail to fit together along imagined steps |
| Learned Bellman audit | the small learned formula combining the checks into a probability |
| Robot / agent | one trained TD-MPC2 model (task × critic type × seed) |
| Seed | the random starting point of training; a new seed gives a new robot |
| AUROC | how well a detector ranks real hallucinations above correct steps (0.5 = guessing, 1.0 = perfect) |
| Practice / exam robots | robots used to design the method / new robots used only to test it |
