# When to trust imagined policy evaluation: results

## ie_stack_s10 (180 imagined episodes; mean gap |imagined - real| = 2.014)

| Detector | Spearman(score, gap) | gap @25% trusted | @50% | @75% | AURC (lower better; random = mean gap) |
| :-- | :-: | :-: | :-: | :-: | :-: |
| LBA | +0.23 | 1.700 | 1.619 | 1.707 | 1.845 |
| D | +0.31 | 1.126 | 1.773 | 1.960 | 1.505 |
| M | +0.39 | 0.890 | 1.607 | 1.893 | 1.412 |
| B | +0.29 | 1.427 | 1.564 | 1.880 | 1.752 |
| oracle | +1.00 | 0.221 | 0.548 | 1.076 | 0.696 |

| Policy evaluation from | Spearman over policies | Mean abs. error |
| :-- | :-: | :-: |
| all episodes | +0.94 | 0.256 |
| LBA | +0.83 | 0.769 |
| D | +1.00 | 0.353 |
| M | +0.94 | 0.308 |
| B | +0.77 | 0.488 |
| oracle | +0.94 | 0.434 |

## ie_ycb_s10 (180 imagined episodes; mean gap |imagined - real| = 1.378)

| Detector | Spearman(score, gap) | gap @25% trusted | @50% | @75% | AURC (lower better; random = mean gap) |
| :-- | :-: | :-: | :-: | :-: | :-: |
| LBA | +0.46 | 0.641 | 0.792 | 1.183 | 0.817 |
| D | +0.41 | 0.680 | 0.994 | 1.306 | 0.893 |
| M | +0.47 | 0.429 | 0.819 | 1.205 | 0.810 |
| B | +0.42 | 0.590 | 0.916 | 1.249 | 0.878 |
| oracle | +1.00 | 0.061 | 0.217 | 0.620 | 0.381 |

| Policy evaluation from | Spearman over policies | Mean abs. error |
| :-- | :-: | :-: |
| all episodes | +0.94 | 0.792 |
| LBA | +0.94 | 1.248 |
| D | +0.83 | 0.832 |
| M | +0.94 | 1.081 |
| B | +0.94 | 1.181 |
| oracle | +0.94 | 0.783 |



[exited with code 0]
