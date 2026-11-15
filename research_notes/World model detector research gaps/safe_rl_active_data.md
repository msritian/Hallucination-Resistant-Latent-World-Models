# World-model error signals for safe RL, shielding, exploration, shift detection, and human-in-the-loop (2023-2026): state of the art and gaps for a learned horizon-aware detector

Scope note: about 16 searches and fetches (Nov 2026). Abstracts were read for most papers, and full text for UNISafe and "When World Models Lie". Items marked "abstract only" were not checked in depth. Some arXiv IDs from 2026 (26xx) could only be seen through arXiv abstract pages.

## Q1. Safe MBRL and shielding: how do they handle model error, and is there a learned "the filter is trusting a wrong model" detector?

### Takeaway
Every world-model shield or latent safety filter gives guarantees only relative to the model. Model error is a recognized failure mode. The newest fixes (UNISafe, ACoFi, "When World Models Lie") all use either one-step epistemic uncertainty or one-step observed latent residuals, calibrated with (adaptive) conformal prediction. None predicts, before execution, whether a multi-step imagined safety rollout is hallucinated. The closest 2026 competitor (Cao and Bansal) states explicitly that ensemble disagreement and value-target consistency residuals "can remain small" when the model is wrong. This directly challenges a Bellman-style detector, so we must address it head-on.

### Cited Findings
- Latent shielding uses a learned latent world model to imagine futures and block unsafe actions. AMBS (Approximate Model-Based Shielding, Goodall and Belardinelli) notes that latent shielding suffers from "accumulating errors that arise from the approximation error of the learned model", and it adds safety critics to mitigate this — [AMBS, arXiv 2308.00707](https://arxiv.org/pdf/2308.00707); continuous follow-up: [Leveraging AMBS for Probabilistic Safety Guarantees in Continuous Environments, arXiv 2402.00816](https://arxiv.org/pdf/2402.00816); original latent shielding: [Do Androids Dream of Electric Fences?, arXiv 2112.11490](https://arxiv.org/pdf/2112.11490)
- A 2025 shielding review says shield guarantees are "always relative to the world model", and lists "How can we obtain a reliable world model?" as an open question — [Shields for Safe Reinforcement Learning, CACM/ACM DL](https://dl.acm.org/doi/10.1145/3715958)
- SafeDreamer (ICLR 2024) adds Lagrangian methods to Dreamer's planning and reports near-zero cost on Safety-Gymnasium. Its authors attribute the failure of prior methods to reach zero cost "primarily to model inaccuracies and inadequate sample efficiency". The project page says LAMBDA and Safe SLAC hit a cost floor "due to the inaccuracies of the world models". This is the authors' own claim, not an independent analysis — [SafeDreamer, arXiv 2307.07176](https://arxiv.org/html/2307.07176v3); [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/ece182f93af26c64187ba3f7dfd4309a-Abstract-Conference.html); [project page](https://sites.google.com/view/safedreamer/)
- I found no paper or repo named "Safe TD-MPC2". The closest is MUARL / "Evidential Latent World Models for Safe Model-based RL" (ICLR 2026 listing). It adds evidential deep learning to TD-MPC2 to get epistemic and aleatoric uncertainty from one forward pass, and uses a dual-constraint MPPI planner that penalizes predicted cost and model uncertainty with adaptive Lagrange multipliers (Safety-Gymnasium navigation). Abstract only — [ICLR 2026 virtual page](https://www.iclr.cc/virtual/2026/10018079)
- SPOWL (2026) learns a world model and a safety-aware latent policy, and switches between model-based planning and direct policy execution to handle model error. Only the listing was seen — [Safe Planning and Policy Optimization via World Model Learning, FAIA 2026](https://journals.sagepub.com/doi/10.3233/FAIA251029)
- Latent Safety Filters (Nakamura, [Peters], Bajcsy, RSS 2025) run HJ reachability in a generative world model's latent space from RGB on a Franka (for example, not spilling a bag). The author list differs between sources — [arXiv 2502.00935](https://arxiv.org/pdf/2502.00935)
- UNISafe (Seo, Nakamura, Bajcsy, CoRL 2025, PMLR 305:4442-4472) says latent filters "overconfidently misclassify risky OOD situations as safe". Its uncertainty signal is a separate ensemble of Gaussian next-latent predictors, scored by Jensen-Renyi divergence. It is single-step: u_{t+1}=D(z_t,a_t). It calibrates an OOD threshold with trajectory-level conformal prediction on in-distribution data only, which bounds false positives. Reachability then runs on the augmented state (z, u). Results: Dubins balanced accuracy 0.94 (JRD); block plucking failure 0.41 (no filter), 0.30 (LatentSafe), 0.20 (UNISafe), with 37.7% of actions filtered. Stated limitations: "the separate ensemble may not faithfully capture the world model's true uncertainty", no unknown disturbances assumed, and component-level guarantees only — [arXiv 2505.00779 (HTML)](https://arxiv.org/html/2505.00779v1); [PMLR](https://proceedings.mlr.press/v305/seo25a.html); [project](https://cmu-intentlab.github.io/UNISafe/)
- Other Bajcsy lab follow-ups:
  - AnySafe adapts latent safety filters at runtime, with the constraint specified at test time — [arXiv 2509.19555](https://arxiv.org/pdf/2509.19555)
  - "What You Don't Know Can Hurt You" finds that latent world models miss partially observable safety state, and fallbacks degrade as a result — [arXiv 2510.06492](https://arxiv.org/html/2510.06492v2)
  - Smooth / CBF-style latent filtering — [arXiv 2511.18606](https://arxiv.org/pdf/2511.18606)
  - Latent filters built on pre-trained vision models — [arXiv 2509.14758](https://arxiv.org/pdf/2509.14758)
- ACoFi (Huriot, Tabbara, Sibai, L4DC 2026) applies adaptive conformal inference to the observed errors of a learned HJ safety filter's safety-value predictions. It switches to the safe policy when the uncertainty range suggests the action may be unsafe, and gives an asymptotic miscoverage bound (a soft guarantee). Benchmarks: Dubins and Safety-Gymnasium — [arXiv 2604.18482](https://arxiv.org/abs/2604.18482)
- **Closest competitor:** "When World Models Lie: Adaptive Safety Analysis Under Wrong Imaginations" (John Cao, Somil Bansal, arXiv 2609.34300, Sep 2026, no venue yet).
  - Signal: the one-step latent residual e_t=||z_t - ẑ_t|| between the encoder latent and the one-step prediction. It feeds ACI to build an adaptive uncertainty set, then minimizes the learned HJ value over that set (PGD).
  - Claim: ensemble disagreement and value-target consistency residuals "can remain small even when the world model's predictions deviate from observations". Evidence: the UNISafe JRD score stayed low through a slippery-zone mismatch until collision, and the ACoFi score stayed at zero for several steps after the shift.
  - Benchmarks: slippery Dubins, ManiSkill Franka pick-and-place avoiding a vase (timestep and observation-staleness shifts), and a hardware Franka tomato throw (480 runs).
  - Stated limitations: the method is **reactive** (it needs the real o_t), it reports no multi-step error analysis, it gives no formal guarantees, and its exchangeability assumption can fail — [abstract](https://arxiv.org/abs/2609.34300); [HTML](https://arxiv.org/html/2609.34300)
- Learned-dynamics reachability with conformal bounds:
  - Hashemi, Qin, Lindemann, Deshmukh: data-driven reachability with conformal inference on surrogate models — [arXiv 2309.09187](https://arxiv.org/abs/2309.09187v1)
  - "Conformal Reachability for Safe Control in Unknown Environments" (2026): per-step conformal intervals on dynamics — [arXiv 2602.03799](https://arxiv.org/html/2602.03799v1)
  - Invariant-measure CP (2026) notes that recursive errors accumulate and break exchangeability, and gives horizon-dependent bounds through Lipschitz propagation — [arXiv 2606.31607 (EmergentMind listing)](https://www.emergentmind.com/papers/2606.31607)
  - Joint over-horizon conformal coverage for output trajectories — [arXiv 2604.02173](https://arxiv.org/html/2604.02173v1)
- Krause group: ActSafe (As, Sukhija, Treven, Sferrazza, Coros, Krause, ICLR 2025) is optimistic on epistemic dynamics uncertainty but pessimistic on safety constraints. It has guarantees under GP/regularity assumptions, and a practical visual-control variant was tested on Safety-Gym and RWRL — [arXiv 2410.09486](https://arxiv.org/pdf/2410.09486); [ICLR 2025](https://iclr.cc/virtual/2025/poster/29177)

### Inferences
- The field has converged on "uncertainty or residual → conformal threshold → pessimistic HJ/shield". The uncertainty signal is always one-step: either predicted by an ensemble before acting (UNISafe) or observed after acting (Cao and Bansal, ACoFi). No method gives a calibrated, **pre-execution, horizon-indexed** score of whether the imagined safety rollout the filter relies on is wrong. That is exactly what a learned Bellman audit could provide.
- Cao and Bansal's dismissal of "value-target consistency residuals" most likely refers to raw TD residuals used as uncertainty. Our learned detector (AUROC 0.814 vs ensemble 0.728) is a different object, but reviewers will cite this paper. A head-to-head on their ManiSkill vase task, with staleness and timestep shifts, would be a direct and cheap rebuttal.
- Our finding that ensemble disagreement shrinks with horizon gives a mechanism for UNISafe's own stated limitation ("separate ensemble may not capture the WM's true uncertainty"). It also gives a mechanism for Cao and Bansal's observation that JRD stays low until collision.

### Gaps
- No "Safe TD-MPC2" paper found apart from MUARL (abstract only). Whether MUARL's evidential uncertainty grows with horizon is unknown.
- The LAMBDA paper itself was not checked for a model-error failure analysis.
- Cao and Bansal's numeric results sit in figures that were not readable.

## Q2. Exploration and active data collection: does ensemble disagreement miss compounding error?

### Takeaway
Yes. This is now reported independently. "Biased Dreams" (RLC 2026) finds that ensemble disagreement in RSSM latent rollouts tracks local, not global compounding, error, and that rollouts get pulled into "attractor" regions where disagreement shrinks while true error grows. This is the same phenomenon as our finding, but on Dreamer, with no proposed fix and no exploration experiment. Nobody has tested whether disagreement-driven exploration (Plan2Explore) under-samples exactly the regions where long-horizon imagination fails.

### Cited Findings
- Biased Dreams (Berger, Frauenknecht, Trimpe, Leibe, RLC 2026; arXiv Apr 2026): ensemble disagreement "does not reliably reflect global compounding model error" in Dreamer RSSM rollouts. An attractor effect pulls rollouts toward well-supported latent regions where uncertainty shrinks even as they drift from the true dynamics, which can inflate return estimates. The authors frame this as a structural limit of epistemic UQ in latent dynamics. No remedy is given in the abstract — [arXiv 2604.25416](https://arxiv.org/pdf/2604.25416)
- The Intervention Gap in Latent World Models (Vakalis, arXiv 2608.29998, Aug 2026):
  - On TD-MPC2, return falls as operator error on task observables grows, while reward-prediction error stays small and flat.
  - Disagreement ranks error only near training support (summary quotes ρ=0.678 in-support vs ρ=0.027 under environment-random support; not verified in full text).
  - The paper recommends auditing intervention fidelity directly — [arXiv 2608.29998](https://arxiv.org/pdf/2608.29998)
- Neubay (offline MBRL) argues that compounding error over long rollouts makes small ensembles inadequate, and uses N=100 ensembles plus LayerNorm — [arXiv 2512.04341](https://arxiv.org/html/2512.04341v2)
- Plan2Explore uses ensemble disagreement as the intrinsic reward, with no analysis of long-horizon compounding — [arXiv 2005.05960](https://arxiv.org/pdf/2005.05960). Disagreement (Pathak et al., ICML 2019) handles noisy TVs because variance collapses once the ensemble learns the mean of the noise. It was shown on a real Sawyer arm — [PMLR v97](https://proceedings.mlr.press/v97/pathak19a/pathak19a.pdf)
- Noisy-TV and learning-progress line:
  - Learning Progress Monitoring (2025) rewards model improvement rather than error, and is tested on MNIST, 3D maze and Atari, not manipulation — [arXiv 2509.25438](https://arxiv.org/html/2509.25438)
  - Aleatoric-aware curiosity — [arXiv 2102.04399](https://arxiv.org/html/2102.04399v3)
  - A Dreamer study of controllable vs perceptual uncertainty — [arXiv 2507.08210](https://arxiv.org/html/2507.08210v1)
- Bayesian MBRL active exploration for manipulation — [arXiv 2404.01867](https://arxiv.org/pdf/2404.01867) (reference list only seen). ActSafe (Q1) is the Krause-group safe-exploration counterpart.

### Inferences
- Because disagreement collapses inside attractors (Biased Dreams) and shrinks with horizon (ours), disagreement-driven exploration should systematically stop visiting states whose **long-horizon** imagination is wrong. That is a testable prediction, and no paper tests it.
- A learned detector of value or Bellman inconsistency is a candidate intrinsic reward or data-acquisition score that targets compounding error. Since our detector failed to raise task success when used as a penalty, the right metric here is **model-error reduction per environment step** (error vs true rollouts at horizon H) and sample cost, not return.
- Noisy-TV risk: a Bellman-inconsistency signal may also fire on stochastic rewards. This needs a noisy-distractor control.

### Gaps
- Plan2Explore horizon-resolved error and disagreement analyses were not found, which suggests the gap is real.
- The Intervention Gap correlation numbers were taken from a search summary and need checking against the full text.

## Q3. Sim-to-real and distribution-shift detection with world models; triggering adaptation

### Takeaway
Shift detection with world models uses **observed prediction residuals** (CUSUM, OOD thresholds) after the shift has already been experienced. Nothing found uses a learned audit of imagined rollouts to predict, before deployment or early in it, where the model will be wrong, or to decide which data to collect to fix it.

### Cited Findings
- Changepoint-Aware World Models (Everest Yang, arXiv 2609.18950): a DreamerV3 agent detects abrupt dynamics shifts from its own prediction error with online CUSUM against a rolling baseline, then forgets stale replay and refits. The closest prior is MBCD (multivariate CUSUM over an ensemble) — [arXiv 2609.18950](https://arxiv.org/abs/2609.18950)
- Self-adapting robotic agents (DreamerV3) use world-model prediction residuals to detect OOD events and trigger fine-tuning — [arXiv 2603.04029](https://arxiv.org/html/2603.04029)
- Novelty detection in RL with world models — [arXiv 2310.08731](https://arxiv.org/pdf/2310.08731) (abstract-level only)
- Reflexive World Models (NeurIPS 2025) use world-model predictions as reference trajectories for rapid motor adaptation, without explicit detection — [arXiv 2505.15589](https://arxiv.org/html/2505.15589); [NeurIPS poster](https://neurips.cc/virtual/2025/poster/115217)
- Cao and Bansal (Q1) test timestep shifts and observation staleness in ManiSkill and on hardware, a ready-made shift benchmark — [arXiv 2609.34300](https://arxiv.org/html/2609.34300)
- "Modeling Latent Disturbances for Robust Decision-Making in World Models" (2026) — [arXiv 2610.07599](https://arxiv.org/html/2610.07599) (title only seen)

### Inferences
- A residual-based detector tells you that the model is wrong now, while a learned imagined-rollout audit can tell you that the plan you are about to trust is wrong. These are complementary. The novel claim is earlier detection (lead time) and fewer false alarms than one-step residuals or CUSUM under gradual shifts (mass, friction, actuator gain in DMC/ManiSkill).

### Gaps
- No sim-to-real paper found that uses a value-consistency audit. Majumdar and Herbert lab-specific 2025-2026 world-model shift papers were not searched.

## Q4. Human-in-the-loop and asking for help with conformal guarantees

### Takeaway
Conformal ask-for-help exists for LLM planners (KnowNo) and for imitation policies (FIPER, FAIL-Detect, ConformalDAgger). Sirius-RM uses a learned dynamics model plus a failure classifier to request help. No work uses a **world-model hallucination detector** for MBRL/MPC agents to trigger human help under a conformal help budget, and none exploits horizon (for example "ask if a plan of length H is untrustworthy").

### Cited Findings
- KnowNo (Ren et al., CoRL 2023): conformal prediction over LLM plan options, asking for help when the prediction set is not a singleton, with a task-success coverage guarantee — [Robots That Ask For Help, arXiv 2307.01928](https://arxiv.org/abs/2307.01928)
- Sirius-RM (Liu, Dass, Martín-Martín, Zhu, ICRA 2024): a learned dynamics model plus a failure classifier, trained continuously on deployment and intervention data. It queries humans only on predicted errors, and reports +23% (sim) and +40% (real) success — [arXiv 2310.17552](https://arxiv.org/pdf/2310.17552); [project](https://ut-austin-rpl.github.io/sirius-runtime-monitor)
- RLIF (ICLR 2024) treats interventions as reward (-1 on intervention) and beats DAgger variants when experts are suboptimal — [arXiv 2311.12996](https://arxiv.org/pdf/2311.12996); [site](https://rlif-page.github.io/)
- ConformalDAgger (2024) asks for help when its conformal uncertainty interval norm exceeds a threshold, and raises its query rate under expert shift — [arXiv 2410.08852](https://arxiv.org/pdf/2410.08852)
- FIPER (NeurIPS 2025): failure prediction without failure data, using RND-based OOD in the policy embedding plus action-chunk uncertainty, both conformally calibrated on successful rollouts. The alarm fires when both exceed threshold. It is not a world model — [alphaXiv 2510.09459](https://www.alphaxiv.org/abs/2510.09459); [NeurIPS](https://neurips.cc/virtual/2025/poster/119369)
- FAIL-Detect ("Can We Detect Failures Without Failure Data?", 2025): failure detection for imitation policies with conformal thresholds — [arXiv 2503.08558](https://arxiv.org/pdf/2503.08558). It is distinct from "Asking for Help: Failure Prediction in Behavioral Cloning through Value Approximation" (value-based help requests) — [ResearchGate](https://www.researchgate.net/publication/372127134_Asking_for_Help_Failure_Prediction_in_Behavioral_Cloning_through_Value_Approximation). The search tool conflated these two papers, and they should be cited separately.
- World-model-based failure detection:
  - FoMo-FD uses a flow-matching world model for surgical imitation policies, with conformal calibration from successful rollouts — [arXiv 2607.27511](https://arxiv.org/html/2607.27511)
  - World-model failure classification for inspection — [arXiv 2602.16182](https://arxiv.org/pdf/2602.16182)
  - RoboMonitor names conformal help requests only as future work — [arXiv 2609.30715](https://arxiv.org/html/2609.30715)
- Conformal safety monitor for flight testing (prediction model, nearest neighbours and CP) — [arXiv 2511.20811](https://arxiv.org/pdf/2511.20811); learning safety from sparse human feedback with CP — [arXiv 2501.04823](https://arxiv.org/pdf/2501.04823); Conformal Decision Theory (Lekeufack, Angelopoulos, Bajcsy, Jordan, Malik) — [arXiv 2310.05921](https://arxiv.org/pdf/2310.05921)

### Inferences
- A Bellman-audit score is a natural nonconformity score for "the imagined plan is unreliable". Split CP on held-out rollouts labelled by true-vs-imagined error gives a guaranteed help-request rate (false-alarm control). Conformal risk control could bound the missed-hallucination rate at a given help budget. The payoff metric is failures or cost avoided per human query, which fits our finding that the detector does not raise success.
- The imagined-regret replan gate combined with the detector gives a two-level escalation: replan when regret is high, ask a human when even the replanned rollout is flagged.

### Gaps
- No world-model-based conformal ask-for-help for RL/MPC agents was found. A 2025-2026 version may exist under the name "foresight" (it appears in FoMo-FD's references, not retrieved).

## Q5 and Q6. Calibrated monitors for learned dynamics, active groups, and candidate gaps with cheap experiments

### Takeaway
Calibration of learned dynamics in 2024-2026 works per step (CP on one-step residuals, ACI online) or uses Lipschitz or invariant-measure arguments for horizon bounds. A learned, horizon-conditioned error predictor used as the conformal score is missing. The most active groups are Bajcsy (CMU, latent safety filters), Bansal (USC, adaptive latent HJ, "When World Models Lie"), Krause (ETH, ActSafe), Lindemann and Deshmukh (USC, conformal reachability), Trimpe (RWTH, Biased Dreams), and Zhu and Martín-Martín (UT, Sirius).

### Cited Findings
- Active groups and newest work:
  - Bajcsy: [UNISafe](https://arxiv.org/abs/2505.00779), [AnySafe](https://arxiv.org/pdf/2509.19555), [partial observability](https://arxiv.org/html/2510.06492), [smooth latent CBF](https://arxiv.org/pdf/2511.18606)
  - Bansal: [When World Models Lie](https://arxiv.org/abs/2609.34300)
  - Krause: [ActSafe](https://arxiv.org/pdf/2410.09486)
  - Lindemann and Deshmukh: [conformal reachability](https://arxiv.org/abs/2309.09187v1)
  - Trimpe and Leibe: [Biased Dreams](https://arxiv.org/pdf/2604.25416)
  - Sibai: [ACoFi](https://arxiv.org/abs/2604.18482)
  - Zhu: [Sirius-RM](https://arxiv.org/pdf/2310.17552)
- Not searched in this pass: Tomlin, Herbert, Majumdar, Pathak, Boedecker 2025-2026 specific papers.
- Horizon-aware CP: invariant-measure CP with Lipschitz horizon bounds — [arXiv 2606.31607](https://www.emergentmind.com/papers/2606.31607); joint-horizon coverage — [arXiv 2604.02173](https://arxiv.org/html/2604.02173v1); multi-step asymmetric CP reachability thesis (RPI 2025) — [RPI DSpace](https://dspace.rpi.edu/items/c9f100a3-1489-44dd-a6aa-96a71530e1cd)

### Inferences: candidate gaps and cheap first experiments
1. **Pre-execution trust monitor for latent safety filters (strongest).**
   - What's missing: UNISafe, ACoFi and Cao-Bansal use one-step or reactive signals, and Cao-Bansal claim that value-consistency residuals fail.
   - Experiment: a TD-MPC2 safety filter or cost-constrained MPPI on ManiSkill3 pick-and-place with a forbidden region (like their vase task), plus DMC with a cost. Shifts: friction or mass, timestep, observation staleness. Compare the detector-gated fallback with ensemble or JRD (UNISafe-style), one-step residual + ACI (Cao-Bansal-style), and a fixed threshold. Metrics: violations, unnecessary interventions, and lead time before violation. A horizon-resolved AUROC plot (detector rises with H, ensemble falls) is the headline figure.
2. **Conformal help budget for MBRL/MPC agents.**
   - What's missing: no world-model detector drives ask-for-help with guarantees.
   - Experiment: simulate a human as an oracle (expert policy or privileged state) on ManiSkill3. Calibrate a split-CP threshold on the detector score at target help rates of 5/10/20%. Report failures avoided per query against ensemble, FIPER-style RND, and random querying, and check empirical coverage.
3. **Horizon-aware data collection versus disagreement exploration.**
   - What's missing: Biased Dreams shows the attractor problem, but nobody measures its effect on exploration or data acquisition.
   - Experiment: with a fixed environment-step budget in DMC (cheetah, walker, quadruped) and ManiSkill, collect data by Plan2Explore-style disagreement, by detector score, by RND, and randomly. Metric: H-step imagined-vs-true error and Bellman-audit AUROC on a held-out state distribution, not return. Add a noisy-distractor control.
4. **Early shift detection, with lead time, versus residual CUSUM.**
   - What's missing: CAWM and self-adapting agents are reactive.
   - Experiment: gradual and abrupt mass, friction or actuator-gain drift in DMC. Compare detection delay and false-alarm rate of detector-on-imagination against CUSUM on one-step error, and trigger fine-tuning only on alarms. Metric: compute or GPU-hours saved for equal post-shift error.
5. **Conformal horizon selection for safety rollouts.**
   - What's missing: horizon-indexed calibration with a learned score.
   - Experiment: per-H conformal thresholds on the detector to choose the longest trustworthy imagination horizon for the shield. Measure violations against fixed-H shields (AMBS-style).

### Gaps
- Labs not searched: Tomlin, Herbert, Majumdar, Pathak, Boedecker.
- Whether Cao-Bansal's "value-target consistency residual" baseline matches anything like a learned Bellman audit needs a full-text check.
- Not fetched: the Lindemann 2024 multi-step CP paper, LAMBDA's own failure analysis, and the MUARL full text.
