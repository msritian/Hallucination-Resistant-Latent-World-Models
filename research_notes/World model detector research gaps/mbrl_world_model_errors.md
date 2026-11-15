# Detecting and Handling World-Model Errors and Hallucinated Imagined Rollouts in MBRL (2023-2026)

Scope note: searched 2026-11-15. Many of the most relevant hits are 2026 arXiv preprints that have not been peer-reviewed. Pre-2023 work is labelled "foundational". Where a citation is given only as a known arXiv ID and was not re-fetched in this session, it is marked "(ID from prior knowledge, not re-fetched)".

## Q1. Online MBRL: how is model error detected and handled, and why do uncertainty penalties often fail online?

### Takeaway
Online Dyna-style MBRL mostly uses ensemble disagreement to decide *where/when to stop or mask* rollouts (M2AC, MACURA, Infoprop) rather than to penalize reward. Latent-planning agents (TD-MPC2, Dreamer) have almost no explicit model-error detector, only value-variance regularizers. Nobody reports a clean result showing that a better error *detector* turns into a better *online policy*. The closest is Hansen & Wang 2026, which uses a hallucination score as a curiosity signal for data collection, not as a penalty.

### Cited Findings
**Dyna-style rollout control (ensemble-based)**
- Foundational: MBPO uses short branched rollouts justified by a "when to trust your model" time-based bound. MACURA's authors describe MBPO's answer as time-based. — [Janner et al., NeurIPS 2019, arXiv 1906.08253](https://arxiv.org/abs/1906.08253) (ID from prior knowledge); characterisation from [MACURA, arXiv 2405.19014](https://arxiv.org/html/2405.19014v1)
- Foundational: M2AC masks rollouts where ensemble disagreement is high, in "Trust the Model When It Is Confident: Masked Model-based Actor-Critic" (NeurIPS 2020). — [arXiv 2010.04893](https://arxiv.org/abs/2010.04893) (ID from prior knowledge, not re-fetched)
- MACURA (ICML 2024, Frauenknecht, Eisele, Subhasish, Solowjow, Trimpe, RWTH Aachen) asks "Where to trust your model?" and treats model accuracy as a local property. It truncates rollouts using epistemic uncertainty measured as ensemble disagreement, and reports gains in data efficiency on MuJoCo. — [PMLR v235](https://proceedings.mlr.press/v235/frauenknecht24a.html); [arXiv 2405.19014](https://arxiv.org/html/2405.19014v1)
- Infoprop (ICLR 2025, same group): "On Rollouts in Model-Based Reinforcement Learning". It finds that data from standard Trajectory Sampling rollouts diverges quickly from the true distribution. Infoprop separates aleatoric from epistemic uncertainty, tracks accumulated model error along a rollout, and terminates the rollout when that error is too large. It reports state-of-the-art Dyna-style MuJoCo results with much longer rollouts. — [arXiv 2501.16918](https://arxiv.org/html/2501.16918v2); [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/29461)
- A later paper summarises Infoprop as using accumulated uncertainty as a truncation criterion, with a maximum horizon of about 50. — [arXiv 2512.04341](https://arxiv.org/pdf/2512.04341)
- Plan To Predict (P2P; NeurIPS 2022, not 2023; Wu, Yu, Chen, Hao, Zhuo) treats the model as the decision maker and the policy as the "dynamics", so the model learns to foresee multi-step uncertainty. It criticises earlier methods for quantifying uncertainty "passively after the samples were generated", and it names the mismatch between one-step training and multi-step use. — [arXiv 2301.08502](https://arxiv.org/pdf/2301.08502.pdf); [NeurIPS 2022 poster](https://neurips.cc/virtual/2022/poster/55012)
- COPlanner (ICLR 2024): "Plan to Roll Out Conservatively but to Explore Optimistically". It uses ensemble uncertainty in two opposite ways: as a penalty during model rollouts and as an exploration bonus during real-environment action selection. — [arXiv 2310.07220](https://arxiv.org/pdf/2310.07220)
- Adaptive horizon, foundational: AOP (arXiv 1912.01188) sets the planning horizon from value uncertainty, including the value's Bellman error. AdaMVE (arXiv 1912.11206) adapts the horizon per state using a *learned model-error function*. DMVE (arXiv 2009.09593) does dynamic-horizon value expansion. — [AOP](https://arxiv.org/html/1912.01188); [AdaMVE](https://arxiv.org/pdf/1912.11206); [DMVE](https://arxiv.org/pdf/2009.09593)

**Latent-planning agents (TD-MPC2 / Dreamer)**
- TD-MPC2's only uncertainty handling is a test-time planning regulariser, borrowed from Feng et al. 2023, that penalises trajectories with high Q-ensemble variance. — [TD-MPC2, arXiv 2310.16828](https://arxiv.org/html/2310.16828v2) (as summarised in search results)
- TD-M(PC)² (Feb 2025) reports large value-approximation errors and overestimation bias in TD-MPC2 on high-dimensional tasks. It traces these to policy mismatch between planner data and the policy prior, and fixes them with a policy constraint rather than a model-error detector. — [arXiv 2502.03550](https://arxiv.org/html/2502.03550v1)
- DoublyAware (June 2025) adds an aleatoric/epistemic uncertainty decomposition to TD-MPC for humanoid locomotion. A third-party summary reports up to 40% faster convergence; this was not verified. — [arXiv 2506.12095](https://arxiv.org/pdf/2506.12095)
- "Planners Exploit Deterministic Error: Dithered Low-Bit World Models" (preprint, Nov 2026) shows a TD-MPC2 planner drifting toward actions that a persistent quantization error favours. It is a concrete case of a planner exploiting model error. — [arXiv 2611.11780](https://arxiv.org/abs/2611.11780)
- EfficientTDMPC (2026) increases the rollout horizon from 3 to 6. Its premise is that return-estimate error from all components compounds with rollout depth. — [arXiv 2605.16692](https://arxiv.org/html/2605.16692)
- MAD-TD (ICLR 2025; Voelcker, Hussing, Eaton, Farahmand, Gilitschenski) uses a small fraction of model-generated data to fix value generalisation to unseen on-policy actions at high update-to-data ratios. This is a *positive* use of imagined data that does not need a detector. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b292ef48c211e2d8288146ff045dd61d-Abstract-Conference.html); [arXiv 2410.08896](https://arxiv.org/html/2410.08896v2)
- Dreamer 4 (Hafner, Yan, Lillicrap, arXiv 2509.24527, Sep 2025) trains agents inside a scalable transformer world model using a "shortcut forcing" objective. It is the first agent to obtain diamonds in Minecraft from offline data alone. The alphaXiv summary says it is more coherent than Oasis and Lucid, which "suffer from hallucinations". — [arXiv 2509.24527](https://arxiv.org/pdf/2509.24527); [project page](https://danijar.com/project/dreamer4/); [alphaXiv overview](https://www.alphaxiv.org/overview/2509.24527v1)

**Why penalties fail online (theory and indirect evidence)**
- Offline pessimism penalties are by construction the sign flip of online exploration bonuses (PEVI). In offline RL, uncertainty is used to rule out poorly visited actions; online, those actions should be favoured. — [Jin, Yang, Wang, "Is Pessimism Provably Efficient for Offline RL?", ICML 2021](https://proceedings.mlr.press/v139/jin21e.html); [Li et al., "Settling the sample complexity of model-based offline RL"](https://arxiv.org/pdf/2204.05275)
- A 2026 preprint argues that the robustness-exploration trade-off changes over training, and that the effect of pessimism shrinks as data accumulate. It studies model-free BR-MDPs, not Dyna. — [arXiv 2605.24345](https://arxiv.org/pdf/2605.24345)
- COPlanner's design, penalty inside the model but bonus in the real environment, implicitly concedes that a pure penalty discourages the data collection that would fix the model. — [arXiv 2310.07220](https://arxiv.org/pdf/2310.07220)

### Inferences
- The user's 7 failed attempts (reward penalty, truncation, and others) match a pattern visible in this literature. Online, errors in imagined data are self-correcting once the agent visits those states. A penalty therefore mainly suppresses exploration, and truncation mainly shortens imagination, which TD-MPC2 already keeps short (horizon 3-5). Infoprop and MACURA show gains with *Dyna-style SAC on MuJoCo*, where rollouts are long and model data dominates the replay buffer. TD-MPC2 is not that regime.
- The detector's information is more likely to pay off where it directs *data collection* (an exploration bonus or targeted data), *planner trust* (horizon or commitment), or *offline / frozen-model* settings where errors are not self-correcting.

### Gaps
- No paper found that runs MACURA or Infoprop style truncation inside TD-MPC2 or Dreamer latent planning. Whether it helps there is open.
- No systematic negative-results paper on online uncertainty penalties for latent planners was found. The "penalties fail online" claim is supported by theory and design choices, not by a dedicated ablation study.
- Could not identify the method "RAD" that the user mentioned. It was not searched successfully and needs a specific title.
- BMPO (Lai et al., ICML 2020, bidirectional rollouts) was not re-fetched.

## Q2. Offline MBRL: ensemble-uncertainty failures, horizon evidence, and Bellman-based uncertainty (MOBILE vs ours)

### Takeaway
Lu et al. 2022 showed that the common ensemble penalties are poorly matched to theory and correlate better with dynamics error than with distribution shift. Two 2026 preprints give direct evidence for the user's core finding: ensemble disagreement *does not grow* with compounding error over long latent rollouts and can even *shrink* for out-of-distribution starts. MOBILE's "Bellman inconsistency" is a variance of Bellman targets across dynamics-ensemble members, not a learned audit, so it inherits these same ensemble pathologies.

### Cited Findings
- Foundational: MOPO (arXiv 2005.13239), MOReL (arXiv 2005.05951), COMBO (arXiv 2102.08363), and RAMBO (arXiv 2204.12581) are the canonical ensemble-penalty, pessimistic-MDP, conservative-value, and adversarial-model approaches. — IDs from prior knowledge, not re-fetched: [MOPO](https://arxiv.org/abs/2005.13239), [MOReL](https://arxiv.org/abs/2005.05951), [COMBO](https://arxiv.org/abs/2102.08363), [RAMBO](https://arxiv.org/abs/2204.12581)
- Lu, Ball, Parker-Holder, Osborne, Roberts, "Revisiting Design Choices in Offline Model-Based RL" (ICLR 2022 spotlight), found:
  - theory calls for penalties bounded by the total-variation distance to the true dynamics, but practice uses heuristic stand-ins;
  - uncertainty "is more correlated with dynamics error than distribution shift";
  - successful penalties "can in fact assign high certainty to data far away from the offline data";
  - longer rollouts plus larger penalties often help;
  - hyperparameters tuned by Bayesian optimisation beat hand-tuned defaults.
  — [arXiv 2110.04135](https://arxiv.org/pdf/2110.04135); [ICLR 2022](https://iclr.cc/virtual/2022/spotlight/6700)
- MOBILE (ICML 2023; Sun, Zhang, Jia, Lin, Ye, Yu, Nanjing University) estimates uncertainty as the disagreement among Bellman estimates Q-hat(s,a) = r_i + γ E V(s'_i) computed across an ensemble of learned dynamics models. It argues this tracks true Bellman error better than MOPO's next-state-variance penalty, which is a loose bound. It is evaluated on D4RL and NeoRL. — [PMLR v202](https://proceedings.mlr.press/v202/sun23q.html); [code](https://github.com/yihaosun1124/mobile)
- LEQ (ICLR 2025; Park & Lee) applies lower-expectile regression to λ-returns from model rollouts, which gives conservatism without explicit uncertainty penalties. It reports the first model-based offline RL method to beat model-free methods on long-horizon AntMaze. — [arXiv 2407.00699](https://arxiv.org/html/2407.00699v2); [ICLR 2025](https://iclr.cc/virtual/2025/poster/29841)
- "Long-Horizon Model-Based Offline RL Without Explicit Conservatism" (arXiv 2512.04341, Dec 2025) argues that compounding error over long rollouts amplifies posterior inaccuracy, so small ensembles are inadequate. It uses N=100 members plus LayerNorm. — [arXiv 2512.04341](https://arxiv.org/html/2512.04341)
- **Biased Dreams** (Berger, Frauenknecht, Trimpe, Leibe; arXiv 2604.25416v2, 2026; RWTH Aachen group, venue not stated). On RSSM and Cat-RSSM (DreamerV2/V3 style) models, 4 DMC tasks, 5 seeds, H=50, with an M=5 external latent ensemble:
  - ensemble disagreement captures local epistemic uncertainty but "does not reliably reflect global compounding model error accumulated over prolonged RSSM latent rollouts";
  - for OOD start states, uncertainty "starts high and quickly converges to ID levels" while proprioceptive error keeps growing, which the authors call an *attractor* effect;
  - closed-loop prior rollouts "systematically overestimate rewards".
  The authors propose no fix and suspect the cause is the inductive bias of VAE-style latents. TD-MPC is not tested. — [arXiv 2604.25416](https://arxiv.org/html/2604.25416v2)
- **Right Ranking, Wrong Scale** (Joyjeet Singh, independent researcher; arXiv 2611.02699, Nov 2026, not peer-reviewed). It tests the RWM-U ensemble (a MOPO-style penalty on the Robotic World Model):
  - disagreement *ranks* error well (Spearman +0.90 at 100 steps) but under-scales it: ±1σ coverage is 16% at 1 step and 4.6% at 100 steps, against 68% nominal;
  - spread fails to grow because every member is fed the ensemble mean; rolling members independently raises coverage to 33%;
  - the members share 89% of their parameters.
  — [arXiv 2611.02699](https://arxiv.org/html/2611.02699); RWM: [arXiv 2501.10100](https://arxiv.org/abs/2501.10100); RWM-U: [arXiv 2504.16680](https://arxiv.org/abs/2504.16680)

### Inferences
- **MOBILE vs the user's Bellman audit.** MOBILE measures *spread across dynamics-ensemble members* of the one-step Bellman target at the current (s,a). It is still an ensemble-disagreement quantity, computed in value units rather than state units. The user's audit is a *learned* detector of Bellman/value inconsistency along the imagined trajectory. It is supervised by ground-truth hallucination labels, or by real-vs-imagined consistency, and it is horizon-indexed. So MOBILE should inherit the "shrinking alarm" pathology that Biased Dreams reports: if all members collapse onto the same attractor, their Bellman targets agree. The user's MOBILE baseline result (0.728 vs 0.814 AUROC) is consistent with this.
- The user's finding that "ensemble alarms shrink and Bellman checks grow with horizon, 15/15 robots and 30/30 DMC tasks" corroborates Biased Dreams' qualitative attractor finding (4 DMC tasks, RSSM only). It extends that finding to (a) TD-MPC2's decoder-free latent space, which rules out VAE inductive bias as the sole explanation, (b) ManiSkill3 manipulation, and (c) a quantitative, cross-task test. This is a publishable positioning point.
- Right Ranking, Wrong Scale suggests one mechanism for the shrinking spread: mean-fed, shared-trunk ensembles. TD-MPC2's Q-ensemble shares the latent dynamics, so it could show the same effect.

### Gaps
- No peer-reviewed paper yet quantifies disagreement vs true error *as a function of horizon* across many tasks for TD-MPC2 latents. Only the two 2026 preprints above exist, on RSSM and one quadruped.
- No paper found that compares a *learned* Bellman-consistency auditor against MOBILE-style ensemble Bellman variance.

## Q3. Value-equivalence, decision-aware models, and critic-model consistency

### Takeaway
Decision-aware MBRL trains the model so that its *value* errors are small (VAML, VaGraM, value equivalence, TD-MPC's latent consistency plus TD loss), but none of these methods produces an *auditor* that reports, at test time, when a particular imagined trajectory's value is untrustworthy. The objective-mismatch literature is now being restated for world-model *evaluation* (2026 position paper).

### Cited Findings
- Wei et al., "A Unified View on Solving Objective Mismatch in Model-Based RL" (TMLR 2024) is a survey. It groups methods into distribution correction, control-as-inference, value equivalence, and differentiable planning, and unifies them as "value optimization-equivalence". — [arXiv 2310.06253](https://arxiv.org/html/2310.06253v2); [TMLR](https://mlanthology.org/tmlr/2024/wei2024tmlr-unified)
- Foundational: objective mismatch (Lambert et al. 2020, arXiv 2002.04523); VAML (Farahmand et al. 2017); VaGraM (Voelcker et al., ICLR 2022, arXiv 2204.01464, value-gradient-weighted model loss); the value equivalence principle (Grimm et al., NeurIPS 2020, arXiv 2011.03506). — IDs from prior knowledge, not re-fetched: [Lambert](https://arxiv.org/abs/2002.04523), [VaGraM](https://arxiv.org/abs/2204.01464), [Grimm](https://arxiv.org/abs/2011.03506); background in the [MBRL survey, arXiv 2206.09328](https://arxiv.org/pdf/2206.09328)
- TD-MPC trains through multi-step latent rollouts with a TD objective, which reduces compounding error, and ramps the planning horizon up from 1 to H. — [TD-MPC, arXiv 2203.04955](https://arxiv.org/pdf/2203.04955)
- "How Should World Models Be Evaluated for Embodied Decision-Making? A Decision-Making-Centric Position" (arXiv 2606.15032, Jun 2026) argues that FID, FVD, and PSNR metrics recreate the objective mismatch problem. — [arXiv 2606.15032](https://arxiv.org/pdf/2606.15032)
- TD-M(PC)² documents that TD-MPC2's value can be badly overestimated while performance stays good. Value error and model error are therefore entangled in latent planners. — [arXiv 2502.03550](https://arxiv.org/html/2502.03550v1)

### Inferences
- In a value-equivalent model such as TD-MPC2, which has no decoder, "hallucination" can only be defined in value or Bellman terms. This makes a Bellman-consistency auditor the *natural* detector for this model class, and is a strong framing argument.
- The user's imagined-regret quantity IR = Q(z,π(z)) − Q(z,a_plan) across heads is a decision-aware disagreement signal: a policy-vs-planner gap. No surveyed method uses that gap to gate open-loop commitment.

### Gaps
- No work found that *learns* a critic-model consistency auditor post hoc on a frozen value-equivalent model.
- Not re-verified: whether any 2025-2026 VaGraM successor (from Voelcker or Farahmand) addresses test-time trust.

## Q4. World-model evaluation: benchmarks and metrics for reliability and hallucination over horizon

### Takeaway
Hallucination detection is booming for *pixel/video* world models: Hansen & Wang 2026, MEND, HaWMPO, WorldRoamBench. There is still no standard benchmark for *latent, decoder-free* world-model reliability over horizon, and no benchmark scores detectors by their downstream decision value.

### Cited Findings
- **Hansen & Wang, "Hallucination in World Models is Predictable and Preventable"** (UC San Diego; arXiv 2606.27326, Jun 2026; venue not stated). This is a direct competitor from the TD-MPC2 authors.
  - Model: a 350M Dreamer-4-style *pixel* world model trained on MMBench2 (65.6k trajectories, 210 tasks).
  - Taxonomy: perceptual, action-marginalised, and scene-diverging hallucination.
  - Detectors: three label-free predictors (tokenizer round-trip residual, flow instability, inter-seed denoising variance). They reach Spearman ρ≈0.80 against ΔPSNR, and AUROC 0.87-0.94.
  - Prevention: coverage-aware sampling gives +0.88 dB. Using the residual as a *curiosity reward* for targeted data collection (50 trajectories per unseen task) raises the normalised MPC score from 0.276 to 0.325.
  - Gaps they leave: no horizon-specific analysis, RGB only, CEM planner rather than a learned policy.
  — [arXiv 2606.27326](https://arxiv.org/html/2606.27326)
- **MEND** (Alrasheed, Yazdan Parast, Azam, Akhtar, Univ. Melbourne; Bailey, Monash; arXiv 2609.39182, Sep 2026). It detects latent hallucination without labels, using a score network on a frozen DINO-WM. AUROC is 0.801 on Wall and 0.691 on PointMaze, against a Gaussian-density baseline of 0.648 and 0.631. Latent error grows from 1.77 at depth 1 to 2.83 at depth 9. Its correction gives 3-6% single-step error reduction, and "gains on a downstream controller are left for future work". It covers deterministic navigation only. — [arXiv 2609.39182](https://arxiv.org/html/2609.39182)
- **HaWMPO** (arXiv 2609.09941, 2026) trains a hallucination-aware model on world-model outputs versus real sequences. It scores imagined chunks and down-weights high-hallucination chunks in VLA policy optimisation through a reward penalty. — [arXiv 2609.09941](https://arxiv.org/html/2609.09941)
- RAW-Dream / "Reinforcing VLAs in Task-Agnostic World Models" (arXiv 2605.12334) filters rollouts whose VLM reward is inconsistent across diffusion noise seeds. WoVR (arXiv 2602.13977) aims to control hallucination in video world models directly. "Perceiving the Knowledge Boundary" (AAAI) uses uncertainty both as an exploration reward and as an imagination filter. — [arXiv 2605.12334](https://arxiv.org/html/2605.12334); [arXiv 2602.13977](https://www.arxiv.org/pdf/2602.13977); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39576)
- WorldRoamBench (Alibaba AMAP, arXiv 2606.31672) evaluates action following, drift, physics, and memory over 10-60 s for interactive *video* world models. It finds that visual quality does not guarantee controllability. — [arXiv 2606.31672](https://arxiv.org/html/2606.31672)
- A driving latent-world-model survey (arXiv 2603.09086) says the field "still lacks unified, interpretable, and closed-loop-relevant evaluation protocols" for long-horizon stability. — [arXiv 2603.09086](https://arxiv.org/html/2603.09086v1)
- PAN (arXiv 2511.09057) reports that all models lose consistency with horizon, at widely varying rates. Rollout-Decoded Reconstruction (arXiv 2608.25017) finds that divergence begins at depths teacher forcing never supervises. — [PAN](https://arxiv.org/pdf/2511.09057); [arXiv 2608.25017](https://arxiv.org/html/2608.25017v1)
- Latent planning with LeWorldModel reaches only 7.0% on a hard TwoRoom task. The authors argue that candidate *ordering*, a ranking/decision problem, is a bottleneck separate from dynamics error. — [arXiv 2605.22164](https://arxiv.org/html/2605.22164v1)
- Not opened (title only): "The Intervention Gap in Latent World Models" (Donna Vakalis, arXiv 2608.29998, ~Sep 2026). The PDF was unreadable, so its content is unknown. — [arXiv 2608.29998](https://arxiv.org/pdf/2608.29998)

### Inferences
- The user's assets (AUROC labels across 15 robots, 30 DMC tasks, and horizon curves) could form the first *latent / decoder-free* hallucination-detection benchmark with horizon-indexed metrics. Pixel-space metrics such as PSNR or a tokenizer residual do not exist for TD-MPC2, which has no decoder, so Hansen & Wang's predictors do not transfer directly. A Bellman-based detector fills exactly that slot.
- Every detector paper above stops at detection, or at a weak downstream effect via penalty or filtering. The user's 7 negative results make a valuable, honest data point.

### Gaps
- The search found no benchmark named for latent (non-pixel) world-model reliability. "WorldModelBench" itself was not fetched; it is video-generation-oriented according to prior knowledge, unverified here.
- Not checked: whether Hansen & Wang's predictors were tested on Newt or TD-MPC2 latents.

## Q5. Who is active, and the most recent 2025-2026 papers

### Takeaway
The closest active groups are Hansen/Wang (UCSD; TD-MPC2, Newt, and now hallucination prediction), Trimpe/Frauenknecht (RWTH Aachen; MACURA, Infoprop, Biased Dreams), Hafner (Dreamer 4), Voelcker/Farahmand/Gilitschenski (Toronto; MAD-TD, value-aware models), and Yang Yu's group (Nanjing; MOBILE). Vision-world-model hallucination work is spreading quickly to VLA labs.

### Cited Findings
- Hansen, Su, Wang: "Learning Massively Multitask World Models for Continuous Control" (Newt, arXiv 2511.19584, Nov 2025). It builds on TD-MPC2 and adds MMBench (200 tasks), demonstration pretraining, and 200+ released checkpoints. The model card says the checkpoints are not expected to generalise reliably to unseen tasks without fine-tuning. — [arXiv 2511.19584](https://arxiv.org/pdf/2511.19584); [HF model card](https://huggingface.co/nicklashansen/newt)
- Hansen & Wang: hallucination prediction and prevention (arXiv 2606.27326, Jun 2026). — [link](https://arxiv.org/html/2606.27326)
- Frauenknecht, Trimpe et al.: MACURA (ICML 2024), Infoprop (ICLR 2025), and Biased Dreams (2026, with Berger and Leibe). — [MACURA](https://proceedings.mlr.press/v235/frauenknecht24a.html); [Infoprop](https://arxiv.org/html/2501.16918v2); [Biased Dreams](https://arxiv.org/html/2604.25416v2)
- Hafner, Yan, Lillicrap: Dreamer 4 (arXiv 2509.24527, Sep 2025). — [link](https://arxiv.org/pdf/2509.24527)
- Voelcker, Hussing, Eaton, Farahmand, Gilitschenski: MAD-TD (ICLR 2025). — [link](https://arxiv.org/html/2410.08896v2)
- Park & Lee: LEQ (ICLR 2025). — [link](https://arxiv.org/html/2407.00699v2)
- Li, Krause, Hutter (ETH): Robotic World Model and its uncertainty follow-up RWM-U. — [arXiv 2501.10100](https://arxiv.org/abs/2501.10100); [arXiv 2504.16680](https://arxiv.org/abs/2504.16680)
- TD-M(PC)² (Feb 2025) and DoublyAware (Jun 2025) are TD-MPC-family error and uncertainty papers. — [TD-M(PC)²](https://arxiv.org/html/2502.03550v1); [DoublyAware](https://arxiv.org/pdf/2506.12095)

### Inferences
- **Highest novelty risk.** Hansen & Wang already own "hallucination is predictable" and "use the predictor for curiosity-driven data collection", in pixel space. The user's differentiation must be explicit: (1) a decoder-free latent world model where pixel predictors are impossible, (2) a *horizon* law (ensemble falls, Bellman rises), and (3) a *learned* Bellman audit that beats MOBILE.
- The Trimpe group is the natural rival or collaborator on the horizon/attractor story.

### Gaps
- Not searched in depth this session: Levine, Finn, Kumar, Amy Zhang, or Boedecker 2025-2026 papers on this exact topic. No relevant hits surfaced in the queries run.

## Q6. Concrete gaps where a horizon-aware learned error detector could be novel, with a cheap first experiment

### Takeaway
The open spaces are where errors do *not* self-correct, or where the detector steers *decisions about trust and data* rather than reward. Ranked by novelty and cost below. All are inferences from the cited gaps.

### Cited Findings (grounding for each gap)
- Ensemble disagreement collapses toward in-distribution levels on OOD latent rollouts (attractor). — [Biased Dreams](https://arxiv.org/html/2604.25416v2)
- Ensemble disagreement ranks error correctly but under-covers it badly, and per-horizon rescaling only partly fixes this. — [Right Ranking, Wrong Scale](https://arxiv.org/html/2611.02699)
- A hallucination predictor used as curiosity improves downstream MPC, in pixel space only. — [Hansen & Wang 2026](https://arxiv.org/html/2606.27326)
- Infoprop and MACURA truncate on accumulated *ensemble* uncertainty. — [Infoprop](https://arxiv.org/html/2501.16918v2); [MACURA](https://arxiv.org/html/2405.19014v1)
- Detector papers leave downstream control to future work. — [MEND](https://arxiv.org/html/2609.39182)
- Released pretrained TD-MPC2-family checkpoints exist (Newt, 200+), with a stated generalisation caveat. — [HF model card](https://huggingface.co/nicklashansen/newt)

### Inferences (proposed directions)
1. **Auditing frozen / pretrained latent world models (offline, no self-correction).** Gap: Newt and TD-MPC2 multitask checkpoints carry an explicit "does not generalise to unseen tasks" warning, and there is no decoder-free tool to say *where and at which horizon* a frozen model fails. Novelty: the first post-hoc, learned Bellman auditor for value-equivalent latent models, with horizon curves, plus a transfer test of whether an auditor trained on seen tasks flags unseen ones. Cheap experiment: take released Newt or TD-MPC2 checkpoints, roll out on held-out MMBench / DMC tasks, label hallucination by real-vs-imagined Bellman error, and report the audit's AUROC against ensemble disagreement and MOBILE as a function of horizon. Training-free on the agent side.
2. **Detector-guided data collection (exploration bonus, not penalty).** Gap: the user's failures all *penalised* imagination. Hansen & Wang's positive result came from treating error as *curiosity*, but only in pixel space and without a horizon dimension. Cheap experiment: in TD-MPC2 online, add a small bonus β·audit(z, horizon h) to the planner's return when *acting* only (COPlanner-style split), with no change to the value-learning targets. Compare with ensemble-disagreement curiosity on 5 sparse or hard ManiSkill3 tasks at 3 seeds. Prediction: Bellman-audit curiosity keeps signal at long horizons where ensemble curiosity decays.
3. **Calibrated, horizon-indexed stopping / commitment (conformal on the audit).** Gap: Right Ranking, Wrong Scale shows that ensembles rank well but are mis-scaled, and no surveyed TD-MPC work gives a *calibrated* guarantee for open-loop commitment. Novelty: conformal prediction on audit score × horizon, giving "commit k steps open-loop with ≤ε hallucination risk". This uses IR as a second feature. Cheap experiment: offline, on saved rollouts, fit per-horizon conformal thresholds on audit or IR. Report coverage versus nominal (direct comparison to Singh's ±1σ coverage table), then compute and success under a latency or replanning budget (open-loop execution between replans).
4. **Replacing ensemble truncation in Dyna-style MBRL where rollouts are long.** Gap: Infoprop and MACURA truncate on ensemble uncertainty, which Biased Dreams shows can collapse. The user's truncation failed inside TD-MPC2, where horizons are already short, but has not been tried in MBPO-style regimes with H=20-50. Cheap experiment: swap Infoprop's accumulated-uncertainty criterion for the learned Bellman audit in an MBPO or Infoprop codebase on 4 MuJoCo tasks. Measure rollout data quality (KL to real rollouts) and return.
5. **Detector as model-training signal (where to spend model capacity), not as policy signal.** Gap: value-aware losses (VaGraM, value equivalence) weight model loss by value sensitivity. None weight it by a *learned, horizon-indexed* failure predictor. Cheap experiment: prioritised replay for the TD-MPC2 world-model loss, where sampling probability ∝ audit score at h=H. Measure the change in real-vs-imagined Bellman error at long horizons, and the return.

### Gaps
- None of these directions has been checked for a 2026 paper that already does it, beyond the searches above. Direction 2 overlaps most with Hansen & Wang 2026. Directions 1 and 3 appear most open based on this session's searches.
- Cost estimates for these experiments in GPU-hours are not sourced. They are judgement calls.
