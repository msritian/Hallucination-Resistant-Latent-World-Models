# Related work notes (2026-10-10)

## Closest to our compounding / anchoring claim
- Biased Dreams (arXiv 2604.25416): ensemble disagreement captures local uncertainty but not compounding error over long
  RSSM rollouts (attractor regions -> false confidence, overestimated returns). Open problem; no fix proposed.
  -> We quantify it in TD-MPC2 (M alarm x.75/.59/.35 at 12/24/40 steps while error x5.7/18/38) and show critic-anchored
  Bellman checks keep pace.
- Long-horizon offline MBRL without conservatism (2512.04341): small ensembles (N=5) inadequate under compounding; uses N=100.
- Operator-on-F (2607.04464): model-level diagnostic comparing k-step latent pushforward with the environment on observed
  quantities; TD-MPC2; catches compounding errors reward metrics miss. Needs real trajectories; scores models, not steps.
- IMPLY (2609.12441): physically anchored consistency (simulator inversion) for video world models; anchored >> self-consistency.
  Supports the anchoring principle; different mechanism and model class.
- Value-consistent representation learning (AAAI 2023); Self-consistent models and values (Farquhar et al. 2021):
  value consistency as a TRAINING signal, not a run-time detector.

## Detection / uncertainty for world models
- Hansen & Wang 2026 (2606.27326): hallucination predictors for pixel world models (decoder round-trip, flow instability,
  seed variance); coverage-aware training; curiosity data collection.
- MOBILE (ICML 2023), MOPO, MOReL, RAVL / edge-of-reach (2402.12527), MACURA (ICML 2024), adaptive truncation (2609.21482),
  HaWMPO (2609.09941), ContactGuard (2608.13438), FARM (2609.11445), ELVIS (2605.04709).
- World models as policy evaluators: WorldEval, WorldGym, dWorldEval, EchoArena, 1X evaluation challenge.
