# Results: grounding 2×2 (critic trained on real vs imagined latents × extra dynamics heads)

Pre-registration: docs/preregistration_grounding.md (with amendment 1). 36 robots: 4 variants (grounded, noaux = grounded
without extra heads, stockaux = stock critic with extra heads, stock) × PushCube, StackCube, PickSingleYCB × seeds
60-62, all trained with the same recipe. Final evaluation clusters through 8691310. Analysis:
`python -m src.tools.grounding_report results/ga_*.tar.gz`. Every claim is reported.

**Inclusion rule:** 35 robots included; ga_stock_ycb_s61 excluded (collection success 0.18 < 0.20).

## Claims (G = success(IR@4) − success(k4))

| claim | result | verdict |
|---|---|---|
| H1 G(real-latent critic) − G(imagined-latent critic) | +0.015 [−0.019, +0.049] | FAIL |
| H2 G(grounded, no extra heads) > 0 | +0.024 [−0.009, +0.057] | FAIL |
| H3 G(grounded) − G(stock) | +0.031 [−0.016, +0.079] | FAIL |

## Success by variant (mean over included robots)

| variant | k1 | k2 | k4 | k6 | random@4 | IR@2 | IR@4 | IR@6 | PE@4 |
|---|---|---|---|---|---|---|---|---|---|
| grounded (9) | 0.713 | 0.706 | 0.619 | 0.508 | 0.568 | 0.676 | 0.628 | 0.588 | 0.646 |
| noaux (9) | 0.672 | 0.662 | 0.568 | 0.496 | 0.533 | 0.634 | 0.592 | 0.506 | 0.601 |
| stockaux (9) | 0.617 | 0.584 | 0.481 | 0.468 | 0.461 | 0.590 | 0.507 | 0.487 | 0.482 |
| stock (8) | 0.682 | 0.623 | 0.547 | 0.479 | 0.508 | 0.630 | 0.525 | 0.471 | 0.519 |

## Reported (not claimed)

- Budget 6: G6(real) − G6(imagined) +0.039 [+0.005, +0.073]; grounded IR@6 − k6 +0.080 [+0.047, +0.113]
  (noaux +0.010, stockaux +0.019, stock −0.007).
- IR@4 − random@4: grounded +0.060, noaux +0.059, stockaux +0.046 (CIs above 0); stock +0.018 [−0.019, +0.054].
  Real − imagined +0.028 [−0.005, +0.062].
- Extra heads: G(aux) − G(no aux) +0.016 [−0.018, +0.050].
- PE@4 − k4: grounded +0.027, noaux +0.033, stockaux +0.001, stock −0.029 (all CIs include 0).
- k1 − k4: grounded +0.094, noaux +0.104, stockaux +0.136, stock +0.135.

## Reading

Grounding is not the decisive switch the Part S result suggested. At budget 4, IR beats fixed chunking by a small,
non-significant margin in every variant except stock, and the grounded-vs-stock difference (+0.031) is not
significant. IR does beat random gating in all three variants with a grounded critic or extra heads, but not in plain
stock TD-MPC2. The one clear effect is at budget 6, and only for the full grounded variant (grounding and extra heads
together: +0.080). That matches the interim look and the frontier study, so the strong IR results so far rest on the
full grounded recipe, and the mechanism is not grounding alone.

Stock models also lose more by committing (k1 − k4 +0.135 vs +0.094), so they have more room to gain. That IR still
does not help there supports the earlier reading: a critic trained on imagined latents is a poor judge of imagined
states. The test it fails is the strict one, though, the comparison with fixed chunks at budget 4.

Consequence for live IR (docs/preregistration_live_ir.md): live IR evaluates the critic on encoded real observations,
not on imagined states, so the stock-set comparison in that study asks directly whether the imagined-state problem is
what holds stock robots back.
