# Meeting Prep Guide: the Whole Story, in Plain Words

*For your own preparation. The short meeting notes are in `docs/meeting_detector_summary.md`; this guide explains everything end to end and lists the figures, videos and likely questions.*

---

## Part 1. The 2-minute version (say this first)

> "Model-based robots plan by imagining the future with a learned world model, and that imagination is sometimes wrong. The model hallucinates outcomes that wouldn't really happen. We asked whether the robot can notice this itself, using only the critic it already has.
>
> The critic estimates how much reward to expect from a situation, and those estimates must fit together from step to step. When the imagination goes wrong, they stop fitting. We turn these consistency checks into about 30 signals and learn, from a little real experience, how to read them, which gives a probability that each imagined step is a hallucination.
>
> Judged against the real simulator, on robots it had never seen, this detector is the best overall of every method we tested, including ensembles of 4 extra world models, for both sudden and gradual hallucinations, with no extra models at all. It's best on Push and YCB; on StackCube, two ensemble methods did better on the new robot, while we were best on the practice StackCube robot."

---

## Part 2. The whole journey, step by step

### Step 1: The starting idea

- **The robot:** TD-MPC2, a model-based reinforcement-learning agent. It has:
  - an **encoder**: turns what it senses into a compact list of numbers (the "latent state");
  - a **world model**: imagines the next latent state for a given action;
  - a **reward model**: predicts the reward;
  - a **critic**: predicts how much reward to expect from here on (5 internal copies);
  - a **planner**: tries hundreds of imagined action sequences and picks the best.
- **The problem:** imagination drifts from reality, especially further ahead. TD-MPC2 therefore only looks 3 steps ahead.
- **Our original ideas:**
  1. **Ground the critic:** train it only on real situations, so it's an independent judge of the imagination.
  2. **Bellman consistency check:** the critic's estimates must satisfy *score now ≈ reward + 0.95 × score next*. If an imagined step breaks this, it's suspicious.
  3. **Trust-weighted planning:** use those checks inside the planner to ignore untrustworthy imagined steps, so the robot can look further ahead.

### Step 2: Building everything

- Implemented the grounded critic, the checks, the baselines (critic disagreement, ELVIS-style, dynamics ensemble, MOBILE-style) and a new planner.
- Set up training on the university GPU cluster (CHTC): containers, CPU-based rendering, checkpointing so jobs survive interruptions.
- Trained robots on several ManiSkill3 tasks, with our critic and with the normal critic, and several "seeds" (independent training runs).

### Step 3: Using the checks inside the planner didn't work (not presented in the meeting)

- **Every** way of putting the trust signal inside the planner's scoring made the robot worse or no better.
- **Why:** the planner searches ~500 candidate plans for the highest score. Any imperfect signal in the score gets exploited: the search finds plans where the signal happens to be wrong. Same reason a perfect-looking exam strategy fails when someone optimises against the grader.
- **Decision:** focus on **detection** itself, and leave "how to use detection in the planner" for later.

### Step 4: Detection looked great, but the test was unfair

- At first we graded the detector using **the critic's own opinion** of what was wrong. Our detector won everything.
- **Problem:** our detector *is* built from the critic, so the critic was grading itself, like a student marking their own exam.
- **Fix:** let the **real simulator** be the judge. Replay the same actions from the same saved moment in the real simulator and compare imagined rewards with real ones.
- **Result:** with the fair judge, the simple version of our detector only **tied** the best expensive method. Good to discover ourselves, before a reviewer did.

### Step 5: Understanding why, then improving the method

- We found **two kinds** of hallucination: **sudden** (breaks at one step) and **gradual** (small errors add up).
- The **one-step check** catches sudden breaks but misses gradual drift.
- So we added the **anchored check**: compare the imagined future with what the critic predicted **at the real starting point**. That catches gradual drift.
- We found the critic has a **small built-in bias**, and that some big signals appear at high-reward moments without real errors. A fixed rule can't separate these from real errors.
- **Solution: the learned Bellman audit.** A small learned reader combines ~31 check signals into a probability, learned from 25 episodes of real experience. We tried two readers (linear and tree-based); **the tree-based one did better** and is our method.

### Step 6: The fair exam

- **Practice robots** (6): used to design and choose.
- **New robots** (6): trained afterwards, never used for any decision. The exam.
- **Result on new robots:** our detector is **best overall on all three cases** (all, gradual, sudden), with **no extra models**. By task: best on Push and YCB; on StackCube, the methods with 4 extra world models did better.

---

## Part 3. The tasks

All tasks are from **ManiSkill3**, a physics-simulation benchmark, with a **Franka Panda robot arm** (7 joints + a two-finger gripper).

| Task | What the robot must do | Episode length | How hard |
| :-- | :-- | :-: | :-- |
| **PushCube** | Push a cube across the table into a goal region | 50 steps (2.5 s) | Easy: trained robots succeed 100% |
| **StackCube** | Pick up a red cube and place it on top of a green cube | 50 steps | Medium: 75–90% success |
| **PickSingleYCB** | Pick up a real-world object (from the YCB set: mugs, cans, fruit, tools; a **different object each episode**) and move it to a goal point in the air | 50 steps | Hard: 30–50% success; new-robot YCB agents were weaker still |

**Common setup:**
- **What the robot senses:** a "state" list of numbers (joint angles and speeds, gripper position, object position and orientation, goal), 45–48 numbers. No camera images.
- **How it acts:** small moves of the gripper (position + rotation change, gripper open/close), 20 times per second.
- **Reward:** dense, between 0 and 1 per step (closer to completing the task → higher).
- **Training length:** Push 200k steps, StackCube 500k, YCB 1M (about 3–12 hours each on the cluster).

**Robots used in the detection experiments:**

| Set | Tasks | Critic | Seed | Count |
| :-- | :-- | :-- | :-: | :-: |
| Practice | Push, StackCube, YCB | ours (grounded) + normal | 10 | 6 |
| Exam | Push, StackCube, YCB | ours (grounded) + normal | 40 | 6 |

Detection works even on robots that aren't good at the task (e.g. YCB). It's about whether the robot's *imagination* matches reality, not whether it succeeds.

---

## Part 3b. The strongest single summary (use this!)

> "Across **10 robots** and 3 tasks, our detector is **best on average** (0.825 vs 0.775 for the learned ensemble), and beats each competitor on **8–10 of the 10 robots**, with no extra models."

The one StackCube robot where we came 3rd (seed 40) was a one-off: on the other 3 StackCube robots we came 1st. Full table: Section 7b of `meeting_detector_summary.md`.

## Part 4. The experiments (detection only)

### Experiment 1: What goes wrong? (your lead's request)

- **How:** freeze a real moment in the simulator; pick 24 actions; let the model imagine the result, and let the simulator show the real result; compare.
- **Action lists:** the robot's own actions (familiar) vs other action lists the model gets asked about (top-ranked and random; unfamiliar).
- **Found:**
  - on familiar actions, imagination is mostly accurate;
  - on unfamiliar ones it drifts 1.5–2× further and becomes **over-optimistic**, most of all for the ones the model rates highest;
  - imagined states don't look "strange" compared with real states (a claim we checked and dropped).
- **Proof figures:** `figs/01_problem_example_ycb.png`, `figs/02_problem_drift_ycb.png`, `figs/03_detector_catches_it_stack.png`.

### Experiment 2: Detection on practice robots

- **How:** for each robot, collect 50 real episodes. From many real moments, imagine 12 steps ahead along the actions actually taken. The simulator says which imagined steps were really wrong. Readers learn from 25 episodes and are tested on the other 25.
- **Score:** AUROC, i.e. how well the detector ranks real hallucinations above correct steps (0.5 = guessing, 1.0 = perfect).
- **Result:** ours 0.835 overall (gradual **0.871**) vs learned ensemble 0.780, best standard method 0.750.

### Experiment 3: The exam on new robots

- **How:** the same as Experiment 2, on 6 robots trained afterwards.
- **Result, like-for-like** (the 3 new robots with our critic, where every method exists): ours **0.849 / 0.844 / 0.884** (all / gradual / sudden) vs learned ensemble 0.830 / 0.836 / 0.837, dynamics ensemble 0.745 / 0.742 / 0.755, MOBILE-style 0.764 / 0.767 / 0.743. Over all 6 new robots (methods that work on any robot): ours 0.866 / 0.860 / 0.897 vs critic disagreement 0.721 / 0.722 / 0.733.
- **By task:** best on Push (0.961) and YCB (0.833); on **StackCube the dynamics ensemble (0.812) and MOBILE-style (0.797) beat us (0.754)**. Say this openly.
- **Proof figures:** `figs/04_result_overall.png`, `figs/05_result_per_task.png`, `figs/06_result_per_robot.png` (all robots); new-robots-only versions are the `figs/backup_new_robots_*` files.

---

## Part 5. What to show, in order

*Files are numbered in presentation order. Everything named `backup_…` is only for questions.*

| # | File | What it shows | What to say |
| :-: | :-- | :-- | :-- |
| 1 | `videos/01_push_trained.mp4`, `02_stack_trained.mp4`, `03_ycb_trained.mp4` | The three tasks, performed by trained robots | "These are our tasks: push a cube, stack two cubes, pick up and move a real-world object." |
| 2 | `figs/01_problem_example_ycb.png` | **The problem.** The model imagines lifting the object to the goal (reward 1.0); in reality it never leaves the table (reward 0.04) | "This is a hallucination: the imagined future looks like success, but reality is failure. The robot can't see this without a simulator." |
| 3 | `figs/02_problem_drift_ycb.png` | **How it happens.** Over 24 steps, imagination drifts from reality and becomes over-optimistic, much more on unfamiliar actions | "On the robot's own actions, imagination is fine. On unfamiliar ones, it drifts and gets too hopeful." |
| 4 | `figs/03_detector_catches_it_stack.png` | **Our signal catches it.** The model 'forgets' the cube is stacked at step 10; our one-step check spikes exactly there, and the anchored check stays high after | "Our Bellman check notices the moment imagination breaks, without seeing reality." |
| 5 | `figs/04_result_overall.png` | **Main result.** Left: all 20 robots, methods needing no extra models. Right: the 10 robots with our critic, every method on the same robots | "Blue with the black outline is ours: the tallest in every group, overall, gradual and sudden. The hatched bars need 4 extra world models; we need none." |
| 6 | `figs/05_result_per_task.png` | **Per task**, averaged over all robots of each task (Push 2, StackCube 4, YCB 4) | "Best on every task. StackCube is harder for everyone; there, sudden errors are a near-tie with the learned ensemble." |
| 7 | `figs/06_result_per_robot.png` | **Robot by robot**, ours vs the two strongest rivals, with ✓/✗ | "Ours is higher on 9 of 10 robots vs the learned ensemble, and 8 of 10 vs the dynamics ensemble." |

**Backup (only if asked):**

| File | When to use it |
| :-- | :-- |
| `figs/backup_new_robots_overall.png`, `backup_new_robots_per_task.png`, `backup_new_robots_per_robot.png` | "How did it do on robots trained after the method was designed?" (6 new robots only. Note: the per-task one shows a single StackCube robot, the one where we came 3rd) |
| `figs/backup_practice_vs_new.png` | "Is it overfitting to the robots you designed on?" (practice vs new robots side by side) |
| `figs/backup_problem_drift_stack.png` | The drift plot for StackCube instead of YCB |
| `videos/backup_*_untrained.mp4`, `backup_ycb_halfway.mp4` | What the robots do before / halfway through training |

## Part 6. Likely questions and good answers

**Q: What exactly is a hallucination in a model without images?**
A: An imagined step whose predictions drift from reality. From the same real moment and the same actions, the imagined rewards differ from what the simulator really gives. We also read out object positions with a simple probe to show it physically.

**Q: Why not just use the simulator?**
A: In the real world there is no simulator. The robot must judge its own imagination. We use the simulator **only to grade** the detector.

**Q: Isn't the critic grading itself?**
A: That was a real problem in our first test, so we switched to the simulator as the judge. All the numbers we present are simulator-judged.

**Q: What's new compared with ensembles or ELVIS?**
A: Ensembles need extra world models (here, 4) and look at disagreement between them. ELVIS looks at value plus uncertainty. We use the critic's **Bellman consistency**: whether the imagined trajectory fits the critic's own predictions, including a check **anchored at the real starting state**, from a critic trained only on real data. It needs no extra models, and it detects better.

**Q: Isn't the learned reader just doing all the work?**
A: We gave the **ensemble the same learned reader**. It still scored lower (0.830 vs 0.866). So the advantage comes from the critic's signals, not from the learning.

**Q: How do you know it isn't overfitting?**
A: We designed the method on practice robots, then tested on brand-new robots trained afterwards. The ranking held. We tried two readers and kept the tree-based one, which did better on both sets; we'll confirm it on one more fresh set.

**Q: How big is the advantage?**
A: Large over standard methods (+0.08 to +0.16 AUROC; e.g. ahead of critic disagreement on all 6 new robots). Small over the learned ensemble (+0.02 overall, +0.05 sudden, about a tie on gradual), while that ensemble needs 4 extra models. Not every task: on StackCube, the 4-model dynamics ensemble was better.

**Q: Does training the critic only on real data matter?**
A: It roughly halves the critic's own noise on correct steps (e.g. StackCube 0.08 vs 0.17), and the anchored check relies on the critic being reliable at the real starting point. Optional to mention: on StackCube it also improved task success (+12–13 points, confirmed with a control experiment), but not on other tasks.

**Q: Why only 3 tasks and these numbers of robots?**
A: Compute and time; each robot takes 3–12 GPU-hours. Next steps are more robots and other benchmarks (e.g. DeepMind Control).

**Q: What would you use the detector for?**
A: Knowing how far ahead the robot can trust its own imagination, and flagging untrustworthy plans. How best to use it in the planner is our next step.

**Q: Why is YCB the hardest?**
A: A different, irregular object every episode; grasping is uncertain; and those robots are weaker at the task, so their imagination is less accurate. All methods score lower there, and ours still leads.

---

## Part 7. Limitations (say these before you're asked)

1. The lead over the learned ensemble is small, and only 3 robots can be compared on it (the ensemble exists only on robots trained with our critic).
2. Not best on every robot: results vary between robots for every method. We lost on 1–2 of 10 robots to individual ensemble methods (StackCube s40, YCB s20), but on average and on most robots we're best.
3. The tree-based reader was chosen after seeing it do better on the new robots too; one more fresh set of robots will settle it.
4. The judge compares predicted vs real **rewards**, so reward-prediction mistakes count as hallucinations too.
5. Three tasks from one benchmark so far.

## Part 8. Next steps

1. Confirm the tree-based reader on another set of new robots, chosen in advance.
2. More robots for tighter error bars.
3. More benchmarks (DeepMind Control).
4. Then: use the detector to decide how far ahead to trust the imagination.

---

## Glossary

| Word | Meaning |
| :-- | :-- |
| Latent state | the robot's compact list of numbers describing the situation |
| World model | the robot's learned imagination of what happens next |
| Critic | network estimating how much reward to expect from a situation (5 internal copies) |
| Grounded critic | our critic, trained only on real situations |
| Bellman consistency check | whether the critic's estimates fit together along imagined steps |
| Anchored check | compares the imagined future with what the critic predicted at the real starting point |
| Learned Bellman audit | our small learned reader that turns ~31 check signals into a probability of hallucination |
| Ensemble | several extra world models trained in parallel; their disagreement signals uncertainty |
| Seed | a separate training run; a new seed gives a new robot |
| AUROC | how well a score ranks real hallucinations above correct steps (0.5 guessing, 1.0 perfect) |
