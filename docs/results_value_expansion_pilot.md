# Value expansion pilot (StackCube, seeds 70/71, 300k steps): results

| Arm | Learning speed (mean eval success after 50k; s70, s71) | Final success (100 eps; s70, s71) |
| :-- | :-: | :-: |
| lba (ours) | 0.207 (0.224, 0.189) | 0.57 (0.46, 0.68) |
| const 0.5 | 0.191 (0.165, 0.216) | 0.70 (0.71, 0.68) |
| none (standard) | 0.164 (0.175, 0.153) | 0.55 (0.52, 0.58) |
| D (ensemble) | 0.133 (0.087, 0.179) | 0.57 (0.46, 0.67) |
| fixed | 0.025 (0.010, 0.039) | 0.22 (0.11, 0.33) |

Pre-registered V1 (lba speed > const and > D) and V2 (lba final >= none) pass, but margins are within noise (2 seeds,
10-episode evaluations). Robust: blind expansion hurts badly; cautious mixing helps learning; lba learns faster than D in
both seeds; const has the best final success. Needs a powered study before any claim.
