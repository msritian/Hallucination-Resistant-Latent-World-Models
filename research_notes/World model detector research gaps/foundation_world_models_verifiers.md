# Foundation / latent world models: hallucination, reliability, verifiers, and gaps for a learned model-error detector

Scope note: research done 2026-11-15 with about 20 search/fetch calls. Items marked "(abstract/snippet only)" were not read in full. Items marked "(training knowledge, not re-verified)" come from prior knowledge, not this session's sources, and should be checked before citing. **Three very close 2026 competitors turned up (Hansen & Wang 2606.27326; "Operator-on-F" 2607.04464; Decision-Level Auditing, NeurIPS 2026 workshop). Read them before claiming novelty.**

## Q1. Latent/feature-space world models for planning: how do they detect or mitigate hallucinated predictions, and do their planners use uncertainty?

### Takeaway
The main pretrained latent-WM planners (DINO-WM, V-JEPA 2-AC, JEPA-WMs, NWM) plan with CEM/MPPI or gradient descent on a plain embedding-distance cost. **None of them uses uncertainty or a reliability check inside the planner.** Several 2025-2026 follow-ups show that these planners do get fooled: decoded "success" that is not real, hallucinated removal of obstacles, and prediction error that does not track planning success. Uncertainty-aware latent MPC exists (ELVIS, using critic-ensemble disagreement), but it is ensemble-based and has not been tested on the pretrained foundation-style WMs.

### Cited Findings
- **DINO-WM** (Zhou, Pan, LeCun, Pinto; NYU + Meta FAIR). Arxiv 2411.04983 (2024; ICML 2025 per training knowledge). It models dynamics on frozen DINOv2 patch embeddings, and the authors said code and models would be open-sourced at dino-wm.github.io — [arXiv HTML](https://arxiv.org/html/2411.04983v1). Whether checkpoints are actually published could not be confirmed this session (see Gaps).
- DINO-WM planned trajectories "more often enter states outside the training distribution," where its decoder cannot reconstruct them. In a Wall example, DINO-WM's decoded final latent showed success while the simulator showed the goal was missed. The fixes proposed (Adversarial World Modeling, Online World Modeling) are training-time changes, not detectors — [Closing the Train-Test Gap in World Models for Gradient-Based Planning, arXiv 2512.09929](https://arxiv.org/pdf/2512.09929) (snippet only).
- A novel distractor object made DINO-WM "hallucinate and erase a critical obstacle." This led a VLM plan verifier to accept an unsafe plan. The proposed fix is test-time observation intervention, not error detection — [Reimagination with Test-time Observation Interventions, arXiv 2506.16565](https://arxiv.org/pdf/2506.16565) (snippet only).
- **V-JEPA 2-AC** (Meta FAIR, arXiv 2506.09985, 2025). It is post-trained on under 62 h of unlabeled DROID video and runs MPC with CEM on an energy, the L1 latent distance from the predicted frame to the goal image. Only the first action is executed before replanning. It works mainly for greedy tasks or tasks with given sub-goals, because the horizon is short. Pretrained checkpoints and an energy-landscape notebook are in facebookresearch/vjepa2 — [EmergentMind summary](https://www.emergentmind.com/topics/action-conditioned-post-training-v-jepa-2-ac); [awesome.ecosyste.ms repo record](https://awesome.ecosyste.ms/projects/github.com%2Ffacebookresearch%2Fvjepa2); [LearnOpenCV guide](https://learnopencv.com/v-jepa-2-meta-world-model-robotics-guide/) (secondary sources).
- **JEPA-WMs**: "What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?" by Terver, Yang, Ponce, Bardes, LeCun (Meta FAIR / INRIA / NYU), arXiv 2512.24497 v2, Jan 2026.
  - Environments: Metaworld Reach / Reach-Wall, Push-T, Wall, PointMaze, Robocasa (zero-shot), DROID plus a real Franka.
  - Planners: CEM, Nevergrad, Adam and GD on an L1/L2 embedding cost. **No uncertainty is used in the planner.**
  - Failure findings: "good rollout accuracy does not guarantee planning success"; arms oscillate around the goal; gradient planners hit local minima; predictions degrade fast when the planning context exceeds the training context.
  - Code, data and checkpoints at github.com/facebookresearch/jepa-wms — [arXiv HTML](https://arxiv.org/html/2512.24497v2).
- **stable-worldmodel (swm)**: Maes, ..., LeCun, Balestriero, arXiv 2605.21800, May 2026. An open platform with WM baselines, planning solvers, and environments that have controllable visual, geometric and physical factors for OOD testing — [arXiv](https://arxiv.org/abs/2605.21800). A search snippet says the paper reports that prediction error correlates poorly with planning success, and that the success and failure distributions overlap even under strong shift. **This was not verified in the abstract** — [arXiv PDF](https://arxiv.org/pdf/2605.21800).
- **Navigation World Models (NWM)**: Bar, Zhou, Tran, Darrell, LeCun (Meta), arXiv 2412.03572, CVPR 2025 (venue from training knowledge). It uses a Conditional Diffusion Transformer (CDiT). It plans by ranking trajectories sampled from NoMaD, using an energy of predicted-vs-goal similarity plus constraint penalties. No uncertainty handling was found — [arXiv](https://arxiv.org/html/2412.03572v2); [Moonlight review](https://www.themoonlight.io/en/review/navigation-world-models).
- **ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC** (arXiv 2605.04709, 2026). Disagreement among an ensemble of latent critics along imagined rollouts sets a time-varying λ, so uncertain far-horizon values count less. That return scores MPPI trajectories — [arXiv PDF](https://arxiv.org/pdf/2605.04709). This is the closest "uncertainty inside the MPPI planner" paper, and it is ensemble-based.
- A bisimulation-based paper reports that DINO-WM degrades under background and distractor changes because it is sensitive to "slow features" — [par.nsf.gov](https://par.nsf.gov//servlets/purl/10674857) (snippet only).

### Inferences
- The standard JEPA/DINO planning stack has no hallucination gate. Several papers show that latent "success" can be false, and that rollout accuracy does not track planning success. A detector that predicts whether a plan's latent outcome is real, conditioned on horizon, plugs directly into an open gap.
- Our result that ensemble disagreement shrinks with horizon matters most here. ELVIS bets on ensemble disagreement for long-horizon gating, so a head-to-head comparison is natural.

### Gaps
- DINO-WM checkpoint availability was not confirmed. From training knowledge, gaoyuezhou/dino_wm on GitHub has code plus released checkpoints and datasets for PointMaze, Push-T, Wall and Rope/Granular; verify this.
- Dreamer 4 (Hafner et al., 2025, "Training Agents Inside of Scalable World Models"), IRIS, DIAMOND and PlaNet were not searched this session. Per training knowledge: DIAMOND (NeurIPS 2024) and IRIS (ICLR 2023) have public code and Atari checkpoints. Dreamer 4 trains a policy in imagination for Minecraft diamonds. None of them has a hallucination detector that I know of. Verify all of this.
- No quantitative per-step V-JEPA 2-AC failure analysis was found.

## Q2. Video/generative world models: known hallucination/physics issues, benchmarks, and any learned "world-model hallucination detector"

### Takeaway
Physics benchmarks consistently show that video WMs fail physics in ways unrelated to how realistic they look. Detection is mostly **offline judging** of finished videos (VLM or fine-tuned judge, IoU metrics). The one serious **online, per-rollout hallucination predictor for an action-conditioned WM** is Hansen & Wang 2026 (TD-MPC2's author). It uses sampling-based signals (round-trip residual, flow instability, inter-seed variance), with no value or Bellman signal and no horizon analysis. Linear probes on video-diffusion hidden states can decode physical plausibility (~81%), but nobody has used them as a rollout-time hallucination detector for planning.

### Cited Findings
- **Hallucination in World Models is Predictable and Preventable**: Nicklas Hansen & Xiaolong Wang, arXiv 2606.27326, 25 Jun 2026.
  - Setup: MMBench2 dataset (427 h, 210 tasks) and a 350M Dreamer-4-style WM (50M encoder, 50M decoder, 250M dynamics) with reward and BC heads.
  - Three hallucination modes: perceptual, action-marginalized ("action ignored"), and scene-diverging.
  - Three label-free predictors, each normalized by per-step latent motion: tokenizer round-trip residual u_r; flow instability across Euler substeps u_f; inter-seed denoising variance u_s.
  - Results: Spearman ρ≈0.80 with rollout ΔPSNR. Per-task AUROC for "action ignored": u_r 0.887, u_s 0.873, u_f 0.868, against a kNN baseline of 0.814. For "scene divergent": u_f 0.939, u_s 0.934, u_r 0.919, against scene-motion 0.927 and kNN 0.731.
  - Uses: coverage-aware sampling and curiosity data collection, with CEM MPC (H=32).
  - What it does not do: use value or Bellman consistency, compare against separately trained ensembles, or analyze horizon dependence beyond 24 frames. TD-MPC2 appears only in related work.
  - Compute: about 58 GPU-days on 8×H100. Code, data and checkpoints are claimed at nicklashansen.com/mmbench2.
  - Sources: [arXiv abs](https://arxiv.org/abs/2606.27326); [arXiv HTML](https://arxiv.org/html/2606.27326v1).
- **Physics-IQ**: "Do generative video models understand physical principles?" (Motamed et al., INSAIT/DeepMind), arXiv 2501.09038, WACV 2026. It has 396 real videos in 66 scenarios: 3 s of conditioning, then a 5 s continuation scored by Spatial IoU, Spatiotemporal IoU, Weighted Spatial IoU and MSE. The best model is VideoPoet (multiframe) at 24.1%, while Sora scores 10.0% despite the best visual realism. Physical understanding is "unrelated to visual realism." The dataset and evaluation code are open — [arXiv](https://arxiv.org/abs/2501.09038); [WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Motamed_Do_Generative_Video_Models_Understand_Physical_Principles_WACV_2026_paper.html).
- **WorldModelBench** (arXiv 2502.20694, NeurIPS 2025 Datasets & Benchmarks). It scores commonsense, instruction following and physics adherence using 67K human labels across 14 models. A fine-tuned 2B judge is 8.6% more accurate than GPT-4o at predicting world-modeling violations. VBench agrees with it on frame quality (r=0.69) but only r=0.28 on physics. The dataset is on HF (Efficient-Large-Model/worldmodelbench) — [arXiv](https://arxiv.org/html/2502.20694v1); [NeurIPS PDF](https://papers.neurips.cc/paper_files/paper/2025/file/4ec03ed08a3fcb59e1c815b5598beff1-Paper-Datasets_and_Benchmarks_Track.pdf).
- **WorldBench** (UCLA, arXiv 2601.21282, Jan 2026). It runs disentangled single-concept physics tests (motion, object permanence, support, scale and perspective) on Kubric scenes. All tested models lack the physical consistency needed for reliable interaction — [arXiv](https://arxiv.org/html/2601.21282v1).
- **Probing for physics.** "The Invisible Hand of Physics: When Video Diffusion Models Know More Than They Show" (NeurIPS 2026, arXiv 2606.05328) inverts the deterministic sampling path. A linear probe on DiT states then separates plausible from implausible videos at about 81% average on IntPhys and InfLevel, beating V-JEPA and VideoMAE. The signal is absent in the VAE latent — [NeurIPS](https://neurips.cc/virtual/2026/poster/149891); [HF papers](https://huggingface.co/papers/2606.05328.md).
- **"Interpreting Physics in Video World Models"** (PMLR v306) describes a mid-depth "Physics Emergence Zone" where physical variables become linearly decodable — [PMLR](https://proceedings.mlr.press/v306/joseph26a.html). "How Do Video Foundation Models Encode Intuitive Physics?" (NeurIPS 2026) finds that physics information peaks at intermediate-to-late depth for V-JEPA, VideoMAE and LTX-Video — [NeurIPS](https://neurips.cc/virtual/2026/175855). Probe directions can steer plausibility judgments — [Causal Physics Steering, arXiv 2605.24322](https://www.alphaxiv.org/abs/2605.24322).
- **"A Unified Definition of Hallucination: It's The World Model, Stupid!"** (arXiv 2512.21577, Dec 2025) defines hallucination as inaccurate internal world modeling that the user can observe. It separates this from planning errors and releases the HalluWorld benchmark (aimed at LLMs) — [arXiv](https://arxiv.org/abs/2512.21577).
- **WorldGym** (Quevedo, Sharma, Sun, Suryavanshi, Liang, Sherry Yang; Stanford/NYU/GDM), arXiv 2506.00613, ICLR 2026. A video WM plus a VLM judge serves as a Monte Carlo policy evaluator. In-WM success correlates highly with real success and preserves policy rankings. The authors admit that realistic object interaction "remains challenging" — [arXiv](https://arxiv.org/html/2506.00613v3); [ICLR](https://iclr.cc/virtual/2026/poster/10008029).
- **WorldEval** (Midea / ECNU), arXiv 2505.19017, 2025. Policy2Vec injects policy actions into a pretrained video model to rank policies and checkpoints, and the result also acts as a "safety detector" for dangerous actions — [arXiv](https://arxiv.org/html/2505.19017v1).

### Inferences
- Hansen & Wang cover the "sampling and reconstruction uncertainty" family well, and in a generative WM. Our Bellman audit is a different signal: decision-relevant value consistency, which grows with horizon. It is also learned, where theirs are label-free heuristics. The honest positioning is that their detectors measure **perceptual/pixel fidelity** (ΔPSNR labels), while ours measures **decision-relevant error** (value or return mismatch). The direct test is to compare the two on the same TD-MPC2 or MMBench2 model.
- The physics probes suggest that a "P(IK)-style probe" for video WMs is feasible. They are still used only for offline plausibility classification, not for gating a planner's imagined rollouts.

### Gaps
- Genie 2/3 (DeepMind), Cosmos (NVIDIA), UniSim, GAIA-1/2 (Wayve), the 1X world model, VBench-2.0, WorldScore and PhyWorld ("How Far is Video Generation from World Model", Kang et al. 2024) were **not searched successfully** this session. From training knowledge, not verified: Genie 3 is not public; Cosmos WFMs have open weights on HF but are large (7B-14B); the 1X WM challenge publishes a dataset. Verify before citing.
- No learned, supervised "hallucination detector" trained on labeled real-vs-imagined outcome mismatches for a video or foundation WM was found, beyond the WorldModelBench judge, which works on whole videos offline.

## Q3. Verifiers / test-time compute for planning with world models, and detecting planner exploitation of model errors

### Takeaway
Test-time scaling with WMs is now active: SWIFT, WEAVER value-head best-of-N, consistency verifiers, and geometric best-of-N, which notably saturates or degrades as N grows. Model exploitation has also become a recognized object, with 2026 theory and repair methods. **But there is no detector that flags, at plan time, that a planner's chosen trajectory is exploiting model error.** The defenses are horizon limits, model repair from failure rollouts, dithering, and execution-history credit.

### Cited Findings
- **SWIFT**: "Can Test-Time Scaling Improve World Foundation Model?" (NVIDIA, COLM 2025, arXiv 2503.24320). Test-time scaling laws hold for video WFMs without retraining — [arXiv](https://arxiv.org/html/2503.24320v1); [NVIDIA](https://research.nvidia.com/labs/aspire/publication/cong.zhu.etal.colm2025/).
- **WEAVER** (arXiv 2606.13672, Jun 2026). A robot WM with a value head does batched best-of-N using latent advantage, with a reported 14% real-world success gain and a 5-10× speedup — [arXiv PDF](https://arxiv.org/pdf/2606.13672) (snippet only).
- **World Action Verifier** (arXiv 2604.01985, Apr 2026). An action-conditioned WM finds its own prediction errors through forward-inverse cycle consistency and uses them for data selection — [arXiv PDF](https://arxiv.org/pdf/2604.01985) (snippet only).
- Consistency-based rollout ranking that "follows similar success-failure trends as learned value predictions" — [arXiv 2605.07514](https://arxiv.org/pdf/2605.07514) (snippet only).
- WAM geometric best-of-N ranks rollouts by cross-view depth reprojection consistency, and **performance can saturate or degrade as N grows** — [arXiv 2607.17454](https://arxiv.org/abs/2607.17454) (snippet only). Degradation at large N is the classic signature of a verifier being exploited (Goodhart).
- MindJourney searches imagined viewpoints with a video WM for VLM spatial reasoning — [arXiv 2507.12508](https://arxiv.org/pdf/2507.12508v2) (snippet only). Temporal Backtracking Search finds that failures cluster early in time, so it branches from the failure point instead of using root best-of-N — [TBS](https://sejoonjun.github.io/TBS/).
- **Imperfect World Models are Exploitable** (Bhamidipaty, Whitammer, Abel, Kochenderfer, Ramamoorthy), arXiv 2605.15960, NeurIPS 2026. It formally defines exploitation as a model preferring policy A over B while the true dynamics prefer the reverse. Exploitation is essentially unavoidable on large policy sets. A relaxed notion admits a **safe horizon**. **No detector is proposed** — [arXiv](https://arxiv.org/abs/2605.15960).
- **Planners Exploit Deterministic Error: Dithered Low-Bit World Models** (arXiv 2611.11780). What matters is error that persists across planner iterations and is shared across candidates, and dithering removes it — [arXiv](https://arxiv.org/html/2611.11780) (snippet only).
- **Reinforced Planning with Latent World Models** (arXiv 2608.18669). A planner trained through a frozen WM exploits its errors. The fix is Dyna-style finetuning on failure rollouts — [arXiv](https://arxiv.org/html/2608.18669) (snippet only). **RENEW** (arXiv 2607.14180) repairs model exploitation from preferences — [arXiv](https://arxiv.org/pdf/2607.14180) (snippet only).
- **DreamLedger** (Li, Tian, Min, Xu, Du), arXiv 2608.23863, Aug 2026. It logs each consumed prediction as a claim, settles it against the execution outcome, and estimates credit from history keyed by condition, region **and horizon**. Below a frozen threshold it refuses the prediction. It cuts burn rate by 4.8 pp vs random refusal (95% CI 0.8-8.7), and was deployed on a Franka — [arXiv](https://arxiv.org/abs/2608.23863).
- **Decision-Level Auditing of Latent Robot World Models** (Rana, Udupi, Rana), NeurIPS 2026 Workshop on Physical World Models. On released JEPA-WM and DINO-WM Push-T planners, it compares the action picked from predicted terminal latents (P) with the action picked from simulator-realized endpoints (O), over 300-action menus.
  - The planners disagree in 337 of 577 cases, yet O does not reliably improve outcomes (ΔG −5.30, p=.0755).
  - A prediction-only near-tie score gives AUROC .714 for selector switches, but it **did not identify higher regret**.
  - Source: [NeurIPS](https://neurips.cc/virtual/2026/165855).
- **Operator-on-F complements value-equivalence: a planning-time diagnostic for latent world models** (arXiv 2607.04464, 2026). On released TD-MPC2 checkpoints, reward-prediction error barely varies across model sizes. The paper proposes measuring Bellman-operator error on the latent predictor at planning time, and its Bellman residual correlates only weakly with return — [alphaXiv](https://www.alphaxiv.org/abs/2607.04464) (snippet only). **This is the closest competitor to our Bellman audit; read it in full.**
- Uncertainty filters for hallucinated imagined rollouts in WM-RL exist (AAAI; IPD, ICML 2026, with a Jensen-Shannon ensemble threshold), and are all ensemble or disagreement-based — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39576/43537); [IPD arXiv 2603.04289](https://arxiv.org/pdf/2603.04289).

### Inferences
- The Decision-Level Auditing paper is a useful negative result: a raw prediction-only ambiguity score finds selector switches but not regret. Our learned detector plus imagined-regret signal targets exactly the missing "predicts regret" piece.
- "Operator-on-F" reports a weak *raw* Bellman residual, while our *learned* audit reaches 0.814 AUROC. The difference to stress is learned vs raw, plus the horizon effect.

### Gaps
- No paper was found that measures detector AUROC for *planner-selected* (argmax) trajectories vs random trajectories, i.e. whether hallucination gets concentrated by optimization. That is an open empirical question.
- No primary results for "process reward model" analogs for robot rollouts were found beyond value-head best-of-N (WEAVER).

## Q4. LLM hallucination-detection methods not yet transferred to world models

### Takeaway
Hidden-state probing has been transferred to video models only for offline physics plausibility. Sampling-consistency, the semantic-entropy analog, has been transferred to generative WMs as inter-seed variance (Hansen & Wang). **Not found for world models:** (a) meaning-level clustering of rollouts, i.e. semantic entropy computed over *decision-relevant outcomes* rather than pixels/latents; (b) a P(IK)-style self-evaluation head trained to predict whether the WM's own rollout will match reality; (c) probes on WM hidden states that predict planning-relevant error (regret or value mismatch) rather than perceptual error.

### Cited Findings
- Semantic entropy: Farquhar, Kossen, Kuhn, Gal, "Detecting hallucinations in large language models using semantic entropy," Nature 2024 — [Nature](https://www.nature.com/articles/s41586-024-07421-0) (training knowledge, URL not re-fetched).
- P(IK) / self-evaluation: Kadavath et al., "Language Models (Mostly) Know What They Know," 2022 — [arXiv 2207.05221](https://arxiv.org/abs/2207.05221) (training knowledge, URL not re-fetched).
- Transfer status found this session:
  - Linear plausibility probes on video DiT states (~81%) — [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/149891).
  - Inter-seed variance in a Dreamer-4-style WM — [arXiv 2606.27326](https://arxiv.org/html/2606.27326v1).
  - No semantic entropy or uncertainty work for video generation surfaced in a targeted search; see the search noted under Q2 for the probing papers.

### Inferences
- Concrete transfer: "semantic entropy over outcomes." Sample K imagined rollouts per candidate plan, cluster them by task-relevant outcome (success/fail, or a value bin), and compute entropy over the clusters. This is cheap with TD-MPC2/DINO-WM if the model is stochastic or MC-dropout is used. One caution: DINO-WM and V-JEPA 2-AC predictors are deterministic, so they would need dropout, augmentation or ensembles.
- Concrete transfer: a "P(IK) head." Train a small head on frozen WM latents (V-JEPA 2-AC or DINO-WM) to predict |realized − predicted| goal distance at horizon h. This is basically our Bellman audit moved to foundation WMs.

### Gaps
- No sources confirmed whether anyone has applied semantic entropy to robot WM rollouts. The absence comes from a limited search, not an exhaustive one.

## Q5. Who is active (2025-2026)

### Takeaway
The active groups are Meta FAIR/LeCun (DINO-WM, V-JEPA 2-AC, NWM, JEPA-WMs, stable-worldmodel), Hansen & Wang at UCSD, which is directly on WM hallucination, theory groups on exploitation (Stanford/Kochenderfer, Edinburgh/Ramamoorthy, Abel), NVIDIA on WFM test-time scaling, and Sherry Yang (NYU/GDM) on WM-as-evaluator.

### Cited Findings
- Meta FAIR / NYU: DINO-WM (Zhou, Pan, LeCun, Pinto) [arXiv](https://arxiv.org/html/2411.04983v1); JEPA-WMs (Terver, Yang, Ponce, Bardes, LeCun) [arXiv](https://arxiv.org/html/2512.24497v2); stable-worldmodel (Maes ... LeCun, Balestriero) [arXiv](https://arxiv.org/abs/2605.21800); NWM (Bar ... LeCun) [arXiv](https://arxiv.org/html/2412.03572v2).
- UCSD: Nicklas Hansen & Xiaolong Wang, WM hallucination prediction with MMBench2 [arXiv](https://arxiv.org/abs/2606.27326).
- Exploitation theory: Bhamidipaty, Whitammer, Abel, Kochenderfer, Ramamoorthy [arXiv](https://arxiv.org/abs/2605.15960).
- WM as evaluator: Sherry Yang group (WorldGym) [ICLR 2026](https://iclr.cc/virtual/2026/poster/10008029).
- NVIDIA: SWIFT test-time scaling for WFMs [NVIDIA Research](https://research.nvidia.com/labs/aspire/publication/cong.zhu.etal.colm2025/).

### Gaps
- Activity from DeepMind Genie, Yilun Du, Berkeley and 1X was not verified this session.

## Q6. Concrete gaps where a horizon-aware learned model-error detector could be novel and feasible (academic GPUs)

### Takeaway
The clearest openings are:
1. A learned, decision-relevant detector benchmarked against the 2026 label-free and ensemble signals.
2. Auditing the released JEPA-WM / DINO-WM / V-JEPA 2-AC planners, with labels from simulator-realized outcomes.
3. A plan-time detector of optimization-induced exploitation, measured as how hallucination rate changes with CEM iterations and best-of-N size.
4. A horizon-indexed hallucination benchmark for latent WMs.
5. P(IK)/semantic-entropy transfers.

Each needs only frozen public checkpoints plus a small head.

### Cited Findings (grounding for each gap)
- **G1. Decision-relevant vs perceptual hallucination; learned vs label-free.**
  - Prior art: Hansen & Wang's predictors target ΔPSNR / action-ignored labels, with no value or Bellman signal and no horizon study — [arXiv HTML](https://arxiv.org/html/2606.27326v1). Operator-on-F finds that a raw Bellman residual correlates weakly with return — [alphaXiv](https://www.alphaxiv.org/abs/2607.04464). ELVIS relies on ensemble disagreement for long-horizon gating — [arXiv](https://arxiv.org/pdf/2605.04709).
  - Missing: a head-to-head of learned Bellman audit vs u_r/u_s/u_f-style signals vs ensembles, as a function of horizon, with return/regret labels.
  - Cheap first experiment: on our TD-MPC2 seed-40 setup, add (i) an inter-seed or MC-dropout variance analog, (ii) the raw Bellman residual (an Operator-on-F-style baseline), and (iii) a 5-member ensemble. Plot AUROC against horizon h=1..H for all of them next to the learned audit.
- **G2. Audit pretrained latent-WM planners on their own benchmarks.**
  - Prior art: JEPA-WMs release code, data and checkpoints, and note that rollout accuracy does not guarantee planning success — [arXiv](https://arxiv.org/html/2512.24497v2). The Decision-Level Audit on JEPA-WM / DINO-WM Push-T found a prediction-only score that does not predict regret — [NeurIPS](https://neurips.cc/virtual/2026/165855). DINO-WM gets false decoded success on Wall — [arXiv 2512.09929](https://arxiv.org/pdf/2512.09929).
  - Missing: a learned detector, trained on (imagined latent, realized latent) pairs, that predicts *regret or goal-miss*.
  - Cheap first experiment: for DINO-WM or JEPA-WM on Push-T/Wall (small sims, one GPU), log CEM candidates, execute them in the sim, label hallucination as realized-vs-predicted goal-distance gap > τ or as success flip, and train an MLP head on predictor features plus a horizon index. Report AUROC by horizon against the Decision-Level near-tie score (.714 on switches) and against embedding-distance baselines.
- **G3. Detecting optimization-induced exploitation.**
  - Prior art: exploitation theory with no detector — [arXiv 2605.15960](https://arxiv.org/abs/2605.15960); best-of-N saturating or degrading in N — [arXiv 2607.17454](https://arxiv.org/abs/2607.17454); persistent shared error being exploited — [arXiv 2611.11780](https://arxiv.org/html/2611.11780); DreamLedger needs execution history — [arXiv 2608.23863](https://arxiv.org/abs/2608.23863).
  - Missing: a plan-time, history-free signal that tells whether the argmax plan is in an exploited region.
  - Cheap first experiment: sweep CEM iterations / elite fraction and best-of-N size N on TD-MPC2 (and DINO-WM). Measure the real hallucination rate of the *selected* plan vs random candidates, and check whether the Bellman audit score rises with optimization pressure while ensemble disagreement does not. Our horizon-shrinking result predicts this outcome.
- **G4. Horizon-indexed hallucination benchmark for latent WMs.**
  - Prior art: existing benchmarks score video output offline (Physics-IQ, WorldModelBench, WorldBench) — [arXiv 2501.09038](https://arxiv.org/abs/2501.09038), [arXiv 2502.20694](https://arxiv.org/html/2502.20694v1), [arXiv 2601.21282](https://arxiv.org/html/2601.21282v1). stable-worldmodel provides controllable OOD factors and baselines — [arXiv](https://arxiv.org/abs/2605.21800).
  - Missing: a benchmark that labels *imagined latent rollouts of action-conditioned planners* for decision-relevant error, per horizon.
  - Cheap first experiment: build on swm environments. Release (rollout, horizon, realized outcome, label) tuples for TD-MPC2, DINO-WM and JEPA-WM, and score detectors on them.
- **G5. P(IK) and outcome-level semantic entropy for WMs.**
  - Prior art: physics plausibility is linearly decodable from video DiT states — [NeurIPS 2026](https://neurips.cc/virtual/2026/poster/149891). No outcome-clustered entropy for WM rollouts was found.
  - Cheap first experiment: train linear probes on frozen V-JEPA 2-AC predictor features to predict realized goal-distance error on a small DROID-sim or Robocasa task. Separately, cluster K sampled TD-MPC2 rollouts by predicted success and compute entropy, then compare its AUROC to the Bellman audit.

### Inferences
- G1 and G3 are the strongest pitches given our existing assets, the horizon finding and imagined regret. G1 is necessary anyway, because reviewers will cite Hansen & Wang 2026 and Operator-on-F. G2 is the most "foundation-model" story and is feasible: the Push-T/Wall sims are light and the checkpoints are public (JEPA-WMs confirmed in its abstract; DINO-WM per training knowledge).
- V-JEPA 2-AC on a real Franka is not feasible for clean labels. Use Robocasa/sim or the JEPA-WMs DROID setup instead.

### Gaps
- Exact GPU cost of DINO-WM / JEPA-WM CEM planning per episode was not found in the sources read.
- Full texts of Operator-on-F (2607.04464), Closing the Train-Test Gap (2512.09929), and the swm results section were not read, so their exact claims should be verified.
