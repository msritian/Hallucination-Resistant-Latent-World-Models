# Robot learning open problems (2023-2026): action chunking / when-to-replan, runtime failure detection, world models as verifiers, and where a learned Bellman-audit / imagined-regret gate fits

Notes compiled 2026-11-15. Most 2026 items are arXiv preprints read at abstract level only (flagged). Items marked [prior-knowledge ID] are well-known papers whose arXiv IDs come from prior knowledge and were not re-fetched this session; verify before citing.

## Q1. Action chunking: what decides when to re-plan, and are there value/critic or world-model criteria?

### Takeaway
The field moved fast in 2026: "adaptive execution horizon" is now a crowded subfield with at least ten preprints, including **world-model-based** (CA3C, DREAM-Chunk, FFDC-WAM) and **critic-based** (ACSAC, DEHP, Decoupled Q-Chunking) criteria. What none of them do (from abstracts) is check whether the world model or critic **is itself reliable** at the horizon being committed. Each one trusts its imagination signal. That is the opening for a horizon-aware model-error detector. Also, a 2026 MIT (Tedrake) paper backs our finding that replanning every step beats chunks of 4: with enough context, the most reactive closed-loop policy wins.

### Cited Findings
**Canonical chunking and smoothing baselines (2023-2025)**
- ACT (Zhao et al., RSS 2023, arXiv 2304.13705) introduced action chunking + temporal ensembling; Diffusion Policy (Chi et al., RSS 2023, arXiv 2303.04137) uses receding-horizon chunk execution; pi0 (Physical Intelligence, arXiv 2410.24164), OpenVLA-OFT (Kim, Finn, Liang, arXiv 2502.19645) use chunked outputs. [prior-knowledge IDs]
- Bidirectional Decoding (BID), Liu, Hamid, Xie, Lee, Du, Finn (Stanford), ICLR 2025, arXiv 2408.17355: samples many chunks every step and picks one by backward coherence and forward contrast. It is a closed-loop resampling rule, not a learned value or WM criterion. [prior-knowledge ID; not re-fetched]
- Real-Time Chunking (RTC): Black, Galliker, Levine (Physical Intelligence / Berkeley), "Real-Time Execution of Action Chunking Flow Policies," NeurIPS 2025. It treats asynchronous chunking as inpainting: actions that will run during inference are frozen and the rest are inpainted. Works on any diffusion/flow VLA without retraining. Evaluated on 12 dynamic Kinetix tasks + 6 real bimanual tasks, and robust to inference delay. RTC handles **latency/smoothness**. It does not decide **whether** to keep committing — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/117747)

**Evidence that open-loop commitment mostly hurts**
- Zeng, Agarwal, Bati, Lee, Ancha, **Tedrake** (MIT), "Revisiting Open-Loop Execution in Robotics: Toward Reactive, Higher-Performing Policies," arXiv 2608.15938 (Aug 2026). Long open-loop execution mostly helps short-context policies imitate **non-Markovian demonstrations**, and this effect is stronger than compounding error. With enough context, "the most reactive closed-loop policies perform best." 4 sim + 2 real tasks — [arXiv 2608.15938](https://arxiv.org/abs/2608.15938)

**Signal-based adaptive horizon (no WM, no critic)**
- DVAC, "Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies," arXiv 2606.03847. Training-free, uses denoising variance. With pi0.5 on LIBERO, success goes from 94.75% to 98.00% while replanning drops 43.0% — [arXiv 2606.03847](https://arxiv.org/abs/2606.03847)
- PACE, "Phase-Aware Chunk Execution," arXiv 2606.00537. Training-free, picks the horizon from the predicted kinematic profile and replans near phase transitions. Beats fixed and adaptive baselines on RoboTwin2.0 with pi0.5 — [arXiv 2606.00537](https://arxiv.org/html/2606.00537)
- "Knowing When to Stop: Adaptive Action Chunking via Internal Cross-Attention Dynamics in VLAs," arXiv 2609.00908 — [arXiv](https://arxiv.org/pdf/2609.00908)
- VLA-Corrector, arXiv 2607.01804. A lightweight detect-and-correct step: keeps long-horizon execution while the chunk is reliable and replans when it is not — [arXiv](https://arxiv.org/pdf/2607.01804)

**Critic / value-based horizon selection**
- ACSAC, "Adaptive Chunk Size Actor-Critic with Causal Transformer Q-Network," Chen et al., arXiv 2605.11009 (May 2026). A causal Transformer critic scores every prefix of candidate chunks, and at each boundary the method picks the chunk size with the highest expected return. Proves a gamma-contraction Bellman operator. OGBench offline and offline-to-online. The abstract does not address critic overestimation — [arXiv 2605.11009](https://arxiv.org/abs/2605.11009)
- DEHP, "Dynamic Execution Horizon Prediction for Chunk-based Robot Policies," Zhao, ..., Shkurti, **Garg**, arXiv 2606.11408. A lightweight horizon-prediction branch trained with online RL on a frozen black-box chunk policy. It learns short horizons in fine stages and long ones in free space. The search snippet says it uses an on-policy V(s) critic; this was not confirmed from the abstract — [arXiv 2606.11408](https://arxiv.org/abs/2606.11408)
- Decoupled Q-Chunking, Qiyang Li (UC Berkeley, Levine group), arXiv 2512.10926. Decouples the critic's chunk length from the policy's, so the policy can run shorter, more reactive chunks — [arXiv](https://arxiv.org/pdf/2512.10926)
- SEAR, "Sample Efficient Action Chunking RL": the critic predicts Q for all chunk prefixes, with random-prefix replanning during data collection — [ResearchGate](https://www.researchgate.net/publication/401494003_SEAR_Sample_Efficient_Action_Chunking_Reinforcement_Learning)
- "Offline RL with Hierarchical Action Chunking," arXiv 2607.20834 — [arXiv](https://arxiv.org/html/2607.20834); "Learning from Mixed-Quality Deployment Experience," arXiv 2609.29000 (an action-chunk critic ranks candidate chunks) — [arXiv](https://arxiv.org/pdf/2609.29000)
- V-GPS (value-guided policy steering, Nakamoto et al., CoRL 2024, arXiv 2410.13816) re-ranks a frozen generalist's chunks with an offline-learned value. [prior-knowledge ID]; described in [No Free Checker survey](https://arxiv.org/html/2609.09250)

**World-model-based horizon / chunk selection (closest competitors to IR gating)**
- **CA3C**, "Commit While Futures Agree: Consequence-Aware Adaptive Action Chunking for Robot Manipulation," Li et al., arXiv 2610.07949 (Oct 2026). An action-conditioned WM imagines several candidate chunks under the same sampling noise. It commits while the imagined futures agree and replans when they diverge, with the horizon set by Bayesian change-point inference. Up to 71.8% relative cut in failure rate. The abstract gives no baselines against value or prediction-error criteria and says nothing on WM reliability — [arXiv 2610.07949](https://arxiv.org/abs/2610.07949v1)
- **"When to Trust Imagination: Adaptive Action Execution for World Action Models"** (FFDC-WAM), Wang, ..., Xiaojuan Qi (HKU), arXiv 2605.06222. A lightweight verifier ("future-reality verification") checks whether the imagined future still matches reality, which sets the effective chunk length. On RoboTwin randomized: +2.54 pp success, 69.10% fewer WAM calls, 33.88% less completion time. On 6 real Unitree G1 tasks: +9.17% success, 74.06% fewer calls, 51.31% less time — [arXiv 2605.06222](https://arxiv.org/abs/2605.06222)
- **DREAM-Chunk**, "Reactive Action Chunking with Latent World Model," Chen, ..., **Yuejiang Liu** (BID author), ..., arXiv 2606.18589. Samples candidate chunks, rolls out latent futures, and runs the chunk whose dreamed state best matches the observed state, switching reactively. Kinetix + 4 tasks on 2 robots. The abstract does not handle WM error — [arXiv 2606.18589](https://arxiv.org/abs/2606.18589)

### Inferences
- The criteria now in use: (a) policy-internal stochasticity (denoising variance, attention, action inconsistency); (b) kinematics/phase; (c) critic value of chunk prefixes; (d) WM imagined-future agreement or future-vs-reality match; (e) learned horizon heads (RL). Our IR gate is in family (c)/(d), but it is distinctive because it is a **regret** (policy action vs. planned action under the ensemble critic) and not an absolute value or a variance.
- FFDC-WAM's "future-reality verification" and DREAM-Chunk's "match dreamed to observed state" are close relatives of our **prediction-error gating** baseline. Our finding that IR ties prediction-error gating except at long commitment on grounded models fits this literature. It also means a pure IR-vs-PE story is not new enough on its own.
- Our 1.73x wall-clock speedup sits next to FFDC-WAM's 69-74% fewer planning calls and DVAC's 43% fewer replans. Compute savings are now an expected metric.
- No paper found makes the gate **aware of how reliable the WM/critic is as a function of horizon**. CA3C's agreement test can be fooled when all imagined futures share the same hallucination (ensemble agreement is not correctness). This is exactly the failure mode the Bellman audit targets.

### Gaps
- Full-text baselines for CA3C, DEHP, DREAM-Chunk were not checked. They may include value baselines I could not see.
- I did not find a paper that compares critic-regret gating against WM-agreement and prediction-error gating on the same benchmark.
- I did not find streaming/async inference papers beyond RTC (e.g., SmolVLA async, "RTC for training-time") in this pass.

## Q2. Runtime failure detection / OOD monitoring: does any use WM imagination to predict failures ahead, and does any connect detection to action with measured task benefit?

### Takeaway
Detection now has a mature first generation (FAIL-Detect, SAFE, Sentinel, FIPER) and a 2026 wave of **world-model-latent monitors** (Foresight, ContactGuard, FARM, PATCH, ReAP) that predict failure from the proposed chunk **before execution**. The consistent gap is that most of them report detection metrics (AUROC, balanced accuracy, conformal FPR), not **task success gained by acting on the alarm**. The verifier survey says this directly. Also, no WM-latent monitor checks whether the WM itself is hallucinating when it scores the chunk.

### Cited Findings
**First generation (2024-2025)**
- FAIL-Detect: Xu et al. (Toyota Research Institute), "Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime Failure Detection for Imitation Learning Policies," RSS 2025, arXiv 2503.08558. Trained on successes only, conformal thresholds, best scores logpZO and RND — [arXiv 2503.08558](https://arxiv.org/pdf/2503.08558); [RSS proceedings](https://www.roboticsproceedings.org/rss21/p073.html); [code](https://github.com/CXU-TRI/FAIL-Detect)
- SAFE (multitask VLA failure detection; Gu et al., arXiv 2506.09937 [prior-knowledge ID]). A lightweight predictor on VLA internal features, time-varying threshold via functional conformal prediction, needs failure labels. The guarantee can break under cross-task shift — [No Free Checker survey](https://arxiv.org/html/2609.09250); [P2 paper](https://arxiv.org/html/2610.31613)
- Sentinel (Agia et al., Stanford/Pavone and Bohg, CoRL 2024, arXiv 2410.04640 [prior-knowledge ID]). Temporal action inconsistency for erratic failures plus a VLM progress judge for task-progress failures — [survey](https://arxiv.org/html/2609.09250)
- FIPER: Römer et al., "Failure Prediction at Runtime for Generative Robot Policies," arXiv 2510.09459. An observation-embedding OOD signal AND action-chunk uncertainty, fit from a few successful rollouts; it alarms when both fire — [arXiv 2510.09459](https://arxiv.org/pdf/2510.09459)

**2026 world-model-latent monitors (predict before execution)**
- **Foresight**, Zhang et al. (Michigan, Princeton, UVA), "Failure Detection for Long-Horizon Robotic Manipulation with Action-Conditioned World Model Latents," arXiv 2606.23085. A frozen V-JEPA 2 encoder + an action-conditioned predictor given the policy's proposed chunk feed a causal-transformer failure scorer. Uses only episode-level labels and functional conformal thresholds. +0.14 balanced accuracy on BEHAVIOR-1K (about 8,557-step tasks) — [arXiv 2606.23085](https://arxiv.org/abs/2606.23085v1)
- ContactGuard, arXiv 2608.13438. Pre-contact monitoring of the single proposed chunk via the predicted future latent — [arXiv](https://arxiv.org/pdf/2608.13438)
- FARM, arXiv 2609.11445. Reads failure signals from the internal predictive states of a frozen robotic WM — [arXiv](https://arxiv.org/pdf/2609.11445)
- PATCH, arXiv 2606.16690. Action-chunk-conditioned latent patch "innovation" (prediction residual) monitoring — [arXiv](https://arxiv.org/pdf/2606.16690)
- ReAP, arXiv 2610.16309. Goes from visual failure recognition to action-conditioned risk prediction — [arXiv](https://arxiv.org/html/2610.16309v1)
- Surgical-robot failure detection via flow-matching world modeling (builds on FIPER), arXiv 2607.27511 — [arXiv](https://arxiv.org/html/2607.27511)
- The survey also lists AHEAD (variance across sampled latent rollouts as uncertainty) and Ward et al. (calibrated latent uncertainty) — [survey](https://arxiv.org/html/2609.09250). Primary sources were not retrieved.

**Other 2026 monitors**
- ActProbe (action-space probe, claims earlier detection than SAFE), arXiv 2606.08508 — [arXiv](https://arxiv.org/pdf/2606.08508); Progress Probes P2, arXiv 2610.31613 — [arXiv](https://arxiv.org/html/2610.31613); RAFAIL (relationship-aware), arXiv 2609.18324 — [arXiv](https://arxiv.org/pdf/2609.18324)
- Streaming Failure Forecasting (Jadhav et al.), arXiv 2611.11688: refits only the monitor's last layer after each episode. 2.18 ms per call on a Jetson CPU core, 434 real runs of 4 VLAs on SO-101. Reports ranking metrics, not task benefit from intervening — [arXiv 2611.11688](https://arxiv.org/abs/2611.11688)
- VLM judges top out around 0.77 on robot success judgment (blog synthesis, secondary source) — [Pebblous blog](https://blog.pebblous.ai/report/robot-vlm-success-judge-ambiguity-bias/en/)

**Connecting detection to action**
- "No Free Checker: A Survey of Verifiers for Robot Policies" (Wan, ..., Linchao Zhu), arXiv 2609.09250. Covers about 150 verifiers (human / rule / learned / model-intrinsic). Central claim: cheaper, denser verifiers are less credible ("no free checker"). For SAFE, Sentinel and FIPER the survey reports detection and calibration **but not task benefit from acting on alarms**. The clearest measured benefit is RoboFAC, whose corrective guidance fed to a running VLA improved real-robot success. Other action links: reachability safety filters, ThriftyDAgger/KnowNo (ask for help), Sirius-Fleet (ahead-of-time failure prediction) — [arXiv 2609.09250](https://arxiv.org/html/2609.09250)
- Latent-space reachability / safety filters in imagination: Nakamura, Peters, Bajcsy, RSS 2025; Gameplay Filters (Nguyen, ..., Fisac, Princeton), CoRL 2025; Kim, Nakamura, Bajcsy, arXiv 2510.06492 (partial observability) — cited in [survey](https://arxiv.org/html/2609.09250)
- Survey open problems: "familiar is not successful" (a familiar-looking failure can still score high), model-intrinsic blind spots (a model gives a high score to failures it does not recognize), and partial observability / shift for conformal and reachability methods — [survey](https://arxiv.org/html/2609.09250)

### Inferences
- "Predict failure before it happens via WM imagination" is **no longer a gap by itself**: Foresight, ContactGuard, FARM and PATCH do this for manipulation.
- The remaining gap is **second-order**: these monitors read WM latents as if they were true. When the WM hallucinates on an OOD chunk (the case where you most need the alarm), the predicted latent can look nominal. This is the survey's "model-intrinsic blind spot." A Bellman-consistency audit of the WM's own imagined rollout is an independent check on whether that imagination can be trusted.
- The second open gap is **closing the loop with task benefit**: alarm, then replan / shorten chunk / ask human, measured as delta success and compute. Our IR-gate results (+0.08 at commitment 6, 1.73x speedup) are already in that form, and that is rare.

### Gaps
- Could not confirm whether Foresight/ContactGuard/PATCH report closed-loop task gains (abstract-level reading only).
- AHEAD and Ward et al. primary sources were not retrieved.

## Q3. World models as verifiers / evaluators at test time, and the trustworthiness of the verifier itself

### Takeaway
World-model policy evaluation (WorldEval, WorldGym, PiL-World, RoboWorld, Cosmos-based evaluators) and WM-based test-time chunk verification (VeriVLA, DreamSteer, DREAM-Chunk) are active. In 2026 several papers documented that **WMs are optimistic**: they predict success for failures, and video WMs "self-correct" failures into successes. Proposed fixes so far are data or training (CureWM, coverage-aware sampling, hallucination-aware training). **Runtime, per-rollout auditing of whether a WM verdict can be trusted at a given horizon** remains open. Notably, Nicklas Hansen (TD-MPC2 author) has a June 2026 preprint showing world-model hallucination is predictable. That is the closest competitor to our framing.

### Cited Findings
**WM as policy evaluator**
- WorldEval: Li, Zhu, Wen, Shen, Xu, "World Model as Real-World Robot Policies Evaluator," arXiv 2505.19017 (2025) — cited in [survey](https://arxiv.org/html/2609.09250)
- WorldGym: Quevedo, Sharma, Sun, Suryavanshi, **Percy Liang**, **Sherry Yang**, "World Model as An Environment for Policy Evaluation," arXiv 2506.00613, ICLR 2026. An autoregressive action-conditioned video WM plus a VLM reward. Success rates within 3.3% of real on average, and the ranking of RT-1-X, Octo and OpenVLA is preserved — [Semantic Scholar](https://www.semanticscholar.org/paper/WorldGym:-World-Model-as-An-Environment-for-Policy-Quevedo-Sharma/8dac59d54f3d268c3467c29fd8ac32d1770404a0); [project](https://world-model-eval.github.io/abstract)
- Scalable Policy Evaluation with Video World Models, arXiv 2511.11520 — [arXiv](https://arxiv.org/html/2511.11520); PiL-World (chunk-wise WM for VLA policy-in-the-loop evaluation), arXiv 2606.05773 — [arXiv](https://arxiv.org/abs/2606.05773); RoboWorld, arXiv 2607.01060 — [arXiv](https://arxiv.org/pdf/2607.01060); GigaWorld-1, arXiv 2607.02642 — [arXiv](https://arxiv.org/pdf/2607.02642); Interactive World Simulator, arXiv 2603.08546 — [arXiv](https://arxiv.org/pdf/2603.08546); Cosmos-Surg-dVRK (NVIDIA Cosmos WFM for surgical policy evaluation), arXiv 2510.16240 — [arXiv](https://arxiv.org/pdf/2510.16240); Cosmos Policy (Kim et al.), arXiv 2601.16163 — per search snippet
- WoVR: "World Models as Reliable Simulators for Post-Training VLA Policies with RL," arXiv 2602.13977 — [arXiv](https://arxiv.org/html/2602.13977v1)

**WM as test-time verifier for VLAs**
- VeriVLA, arXiv 2610.21985. Adaptive test-time verification for flow VLAs: a JEPA latent WM ranks candidate chunks by predicted task progress, and verification only triggers adaptively — [arXiv](https://arxiv.org/html/2610.21985v1)
- DreamSteer, arXiv 2607.02865. A latent WM plus a language-conditioned value model steer a VLA at deployment with no finetuning — [arXiv](https://arxiv.org/html/2607.02865)
- Pre-VLA (ARGUS verifier filters chunks before execution or imagination), arXiv 2605.22446 — [arXiv](https://arxiv.org/html/2605.22446); SafeDojo, arXiv 2606.20698 — [arXiv](https://arxiv.org/html/2606.20698)
- RoboMonkey: Kwok, Agia, Sinha, Foutter, Li, Stoica, Mirhoseini, **Pavone**, CoRL 2025. Samples and perturbs VLA actions and scores them with a VLM-based verifier — [survey](https://arxiv.org/html/2609.09250)
- V-JEPA 2 (Meta, arXiv 2506.09985 [prior-knowledge ID]) does zero-shot planning with an action-conditioned latent WM. It is the backbone used by Foresight — [Foresight](https://arxiv.org/abs/2606.23085v1)

**Trustworthiness of the WM verifier**
- "World Models Dream of Success: Diagnosing and Repairing Failure Insensitivity in Robot World Models," Xu et al., arXiv 2610.09134. Four released checkpoints from two families, tested on 484 held-out LIBERO failure counterfactuals. Optimism (predicting success for failed actions) is 80% after official fine-tuning. CureWM (fine-tuning on verified counterfactual failures) brings it to 30-43%. On a real arm, success-like predictions on failures drop from 90% to 33%. Counterfactual failures give a 0.124 success-failure value gap vs 0.014 for fresh on-policy failures — [arXiv 2610.09134](https://arxiv.org/abs/2610.09134)
- dWorldEval (ICML 2026): video WMs "self-correct" failures into false positives. Proposes a discrete-diffusion WM evaluator — [ICML 2026 poster](https://icml.cc/virtual/2026/poster/65898)
- **"Hallucination in World Models is Predictable and Preventable," Nicklas Hansen & Xiaolong Wang (UCSD), arXiv 2606.27326** (June 2026). Trains a 350M WM on MMBench2 (427 h, 210 tasks, with live simulators). Names three hallucination modes (perceptual, action-marginalized, scene-diverging) and three signals that predict where the model fails. These signals are used as **curiosity rewards for data collection**, not as planning or replanning gates. Claims hallucination is mainly a coverage problem, fixed by coverage-aware sampling and 50-trajectory finetuning — [arXiv 2606.27326](https://arxiv.org/abs/2606.27326)
- HaWMPO, arXiv 2609.09941. A chunk-level hallucination scorer from generated vs observed video, used in WM-based policy optimization. It notes that failure does not imply hallucination — [arXiv](https://arxiv.org/pdf/2609.09941)
- DreamLedger, "Where to Refuse World-Model Imagination Using Execution-Settled Credit," arXiv 2608.23863 — [arXiv](https://arxiv.org/pdf/2608.23863)
- Surveys argue evaluation WMs should be judged by rank consistency and decision reliability, not pixel accuracy. They recommend measuring hallucination rate, rollout drift and policy exploitation of model artifacts — [World Models for Robotic Manipulation survey, arXiv 2606.00113](https://arxiv.org/pdf/2606.00113); [World Model for Robot Learning survey, arXiv 2605.00080](https://arxiv.org/html/2605.00080v1); "How Should World Models Be Evaluated for Embodied Decision-Making?" arXiv 2606.15032 — [arXiv](https://arxiv.org/pdf/2606.15032)
- No Free Checker: a generated rollout can show success where the real robot fails, so the reference "must come from the real outcome" and often needs a human — [survey](https://arxiv.org/html/2609.09250)

### Inferences
- Hansen & Wang (2606.27326) overlaps strongly with our "hallucination detector for TD-MPC2-family WMs" framing. It likely uses the Newt/MMBench2 TD-MPC2 lineage, but this was not confirmed from the abstract. Their predictors drive **data collection**. Ours (the Bellman audit) drives **runtime commitment and replanning**, and that is the differentiator to stress. Read their full paper before any claim of novelty.
- The optimism result (80% of failures predicted as success) means any WM-agreement chunk gate (CA3C, DREAM-Chunk, VeriVLA) will tend to over-commit exactly when the policy is heading into failure. A Bellman-consistency audit (reward/value consistency of imagined transitions, not pixel plausibility) is a natural cross-check, because optimism shows up as a value that is inconsistent with the predicted reward or next value.

### Gaps
- Did not verify whether any WM-evaluator paper gives per-rollout trust scores (as opposed to aggregate rank correlation).
- Cosmos-Policy/Cosmos evaluation details not fetched.

## Q4. Who is active (2025-2026)

### Takeaway
The chunking and failure-monitoring frontier is spread across many groups. Pavone/Stanford (Sentinel, RoboMonkey), Finn/Stanford (BID, OFT), Levine/Berkeley + Physical Intelligence (RTC, Decoupled Q-chunking), TRI (FAIL-Detect), Tedrake/MIT (open-loop revisited), Garg/Shkurti (DEHP), Princeton (Fisac: Gameplay Filters; Foresight co-authors), Hansen/Wang UCSD (WM hallucination), Liang/Sherry Yang (WorldGym), and Meta (V-JEPA 2) are the recognizable names.

### Cited Findings
- Physical Intelligence / Berkeley: RTC (Black, Galliker, Levine), NeurIPS 2025 — [arXiv 2506.07339](https://arxiv.org/abs/2506.07339); Decoupled Q-Chunking (Qiyang Li, Berkeley) — [arXiv 2512.10926](https://arxiv.org/pdf/2512.10926)
- Stanford/NVIDIA Pavone: RoboMonkey (CoRL 2025), Sentinel — [survey](https://arxiv.org/html/2609.09250)
- Stanford Finn: BID (ICLR 2025) [prior-knowledge]; BID first author Yuejiang Liu is a co-author on DREAM-Chunk — [arXiv 2606.18589](https://arxiv.org/abs/2606.18589)
- Toyota Research Institute: FAIL-Detect (RSS 2025) — [GitHub CXU-TRI](https://github.com/CXU-TRI/FAIL-Detect)
- MIT Tedrake: Revisiting Open-Loop Execution — [arXiv 2608.15938](https://arxiv.org/abs/2608.15938)
- Toronto/Garg: DEHP — [arXiv 2606.11408](https://arxiv.org/abs/2606.11408)
- Princeton/Michigan/UVA: Foresight — [arXiv 2606.23085](https://arxiv.org/abs/2606.23085v1); Fisac (Princeton) Gameplay Filters — [survey](https://arxiv.org/html/2609.09250)
- Berkeley Bajcsy / CMU: latent reachability safety filters — [survey](https://arxiv.org/html/2609.09250)
- UCSD Hansen/Wang: WM hallucination — [arXiv 2606.27326](https://arxiv.org/abs/2606.27326)
- Stanford Liang / Sherry Yang: WorldGym (ICLR 2026) — [project](https://world-model-eval.github.io/abstract)
- HKU Xiaojuan Qi: FFDC-WAM — [arXiv 2605.06222](https://arxiv.org/abs/2605.06222)
- NVIDIA: Cosmos-based evaluators — [arXiv 2510.16240](https://arxiv.org/pdf/2510.16240)

### Inferences
- Many key 2026 entries (CA3C, VeriVLA, PATCH, ContactGuard, FARM) come from groups not on the original list. The space is crowded and moving month by month, so novelty claims need a fresh arXiv sweep right before submission.

### Gaps
- No 2026 chunk-gating or failure-monitoring paper was found from NVIDIA GEAR or CMU specifically in this pass.

## Q5. 3-5 concrete gaps where a horizon-aware learned model-error detector or regret gate could be novel, with cheap first experiments

### Takeaway
The defensible gaps are not "use a WM to decide chunk length" (CA3C, FFDC, DREAM-Chunk already do this) or "use WM latents to predict failure" (Foresight et al. already do this). The gaps are **(1) auditing the imagination that these gates and monitors rely on**, **(2) regret rather than agreement as the commit signal, compared head to head**, **(3) closing the loop with measured task and compute benefit**, and **(4) horizon-indexed trust for WM-based policy evaluation**.

### Cited Findings (grounding for each gap)
- Gap A: WM-agreement gates assume the imagination is right. Optimism rates of 80% on failures — [2610.09134](https://arxiv.org/abs/2610.09134); agreement-based commit — [CA3C](https://arxiv.org/abs/2610.07949v1); no WM-error handling stated — [DREAM-Chunk](https://arxiv.org/abs/2606.18589)
- Gap B: critic-based horizon methods (ACSAC, DEHP) report no head-to-head against WM agreement or prediction error, and do not discuss overestimation (abstract level) — [ACSAC](https://arxiv.org/abs/2605.11009); [DEHP](https://arxiv.org/abs/2606.11408)
- Gap C: monitors report detection, rarely task benefit — [No Free Checker](https://arxiv.org/html/2609.09250); [SFF](https://arxiv.org/abs/2611.11688)
- Gap D: WM evaluators are judged by aggregate rank and success-rate match (WorldGym 3.3%), not per-rollout trust — [WorldGym](https://world-model-eval.github.io/abstract); hallucination predictors exist but are used for data collection — [Hansen & Wang](https://arxiv.org/abs/2606.27326)

### Inferences (proposed gaps + cheap experiments)
1. **Audited commitment: gate chunk length on "imagination is still trustworthy" and not only "imagination says it's fine."** What's missing: CA3C/DREAM-Chunk/VeriVLA/FFDC never test whether the WM is hallucinating at step k of the chunk. Novelty: use the Bellman audit score at each imagined step as a horizon-indexed trust bound, and commit up to the first step where audit > threshold. IR can also be combined (commit while IR is small AND the audit passes). *Cheap experiment:* in ManiSkill3 with TD-MPC2 (our seed-40 setup), compare fixed-4/6, PE gating, IR gating, ensemble-agreement gating (a CA3C-style proxy: variance of imagined latents across ensemble/noise) and audit-gated IR. Report success, replans per episode and wall-clock. Key slice: OOD perturbations (object pose / friction shift), where agreement-style gates should over-commit.
2. **Does agreement fail where regret doesn't? A controlled "shared hallucination" test.** What's missing: no paper isolates the case where all imagined futures agree and are all wrong. *Cheap experiment:* make an ungrounded vs grounded TD-MPC2 pair (we already have the 2x2 grounding study). On rollouts where ensemble latent variance is low but the Bellman audit flags error, measure the real outcome after committing. This directly tests whether audit or IR catches what agreement misses. This complements the optimism-rate analysis of 2610.09134.
3. **Imagination-audited failure forecasting with a closed-loop action and task-benefit accounting.** What's missing: Foresight/ContactGuard/PATCH predict failure from WM latents but don't audit the WM, and mostly report detection metrics. *Cheap experiment:* train a FAIL-Detect-style success-only score and a Foresight-style "WM-latent failure scorer" on TD-MPC2 latents, then add the audit score as a feature or gate. Measure AUROC and **lead time** (steps before failure). Then close the loop: on alarm, replan or shorten commitment, and report delta success at matched compute. This follows the survey's call for task benefit.
4. **Transfer the regret gate to a diffusion/flow chunk policy (does IR need a planner?).** What's missing: IR is defined for TD-MPC2's planner vs. policy prior. For a Diffusion Policy, the analogue is Q(s, a_t^{fresh sample}) − Q(s, a_t^{committed chunk}), using a critic learned offline (V-GPS-style) or a TD-MPC2 critic trained on the same ManiSkill3 data. Our Part S result (IR does not transfer to stock TD-MPC2) is a warning: report it honestly as a condition (requires grounded critic). *Cheap experiment:* ManiSkill3 PushCube/PickCube Diffusion Policy (from the ManiSkill baselines) + an offline ensemble Q. Compare fixed chunk 8/16, BID-lite, DVAC-style variance gating and IR gating. Then add the audit gate on the critic (a Bellman residual on its own targets).
5. **Per-rollout trust for WM-based policy evaluation (an "audited WorldGym").** What's missing: WM evaluators are validated on aggregate rank correlation, and per-rollout verdicts are untrusted (dWorldEval self-correction, CureWM optimism). *Cheap experiment:* use TD-MPC2 imagination as an evaluator of checkpoints from a training run in ManiSkill3/DMC. Compute predicted return per checkpoint with and without discarding audit-flagged imagined rollouts, and correlate with true sim return (Spearman over checkpoints, plus the false-success rate). Cheap because the sim gives ground truth for free.

### Gaps
- Novelty versus Hansen & Wang 2606.27326 and CA3C 2610.07949 must be checked from full text, which I could not read here. Both are the highest-priority reads.
- No paper found evaluates adaptive chunking on ManiSkill3 or DMC with TD-MPC2. The benchmarks used are LIBERO, RoboTwin, Kinetix, OGBench and BEHAVIOR-1K. This is a positioning risk (reviewers may ask for LIBERO or RoboTwin with pi0.5), but it is also a niche.
