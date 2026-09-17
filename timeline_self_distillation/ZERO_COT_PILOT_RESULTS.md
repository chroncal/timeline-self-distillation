# Zero-CoT teacher pilot results

Date: 2026-09-17 UTC

This is an implementation and five-seen-image diagnostic, not a generalization
claim. The five frozen Long-CoT trajectories are the same records used by the
earlier timeline experiments. Each reported evaluation uses 16 fixed draws per
image (80 boxes total), invalid boxes count as IoU zero, and no ground-truth box
is used in an update.

## Free teacher quality

The repaired teacher is exactly `C0 + BOX_OPEN`: no generated reasoning, no
entity bridge, and no repeated question. Its free generations were compared to
the unmodified terminal student with matched seeds.

| sample | terminal student IoU | zero-CoT teacher IoU | delta |
|---|---:|---:|---:|
| row-0 | 0.3485 | 0.2743 | -0.0743 |
| row-1 | 0.3828 | 0.5628 | +0.1799 |
| row-3 | 0.8259 | 0.7782 | -0.0477 |
| row-4 | 0.8601 | 0.7124 | -0.1477 |
| row-5 | 0.9212 | 0.6885 | -0.2326 |
| **mean** | **0.6677** | **0.6032** | **-0.0645** |

All 80 boxes from each condition parsed successfully. The zero-CoT teacher had
higher Acc@0.5 (0.6625 versus 0.5875) because it repaired row-1, but its mean
geometry was worse and it damaged four of the five samples.

Across the 80 generated boxes, the terminal student was 0.6677 +/- 0.2569 IoU
and the teacher was 0.6032 +/- 0.2714 (sample standard deviation). The matched
teacher-minus-student delta was -0.0645 +/- 0.2443. These draws are clustered
within only five images, so the dispersion is diagnostic rather than a
generalization confidence interval.

## OPD effect

Only the terminal bbox adapter was trainable. Reasoning was frozen and replayed
in full. Learning rate was 0.001 and training ran for 20 on-policy updates.

| divergence | step | mean IoU | Acc@0.5 | valid boxes |
|---|---:|---:|---:|---:|
| no update | 0 | 0.6677 | 0.5875 | 80/80 |
| reverse KL | 10 | 0.5993 | 0.5250 | 77/80 |
| reverse KL | 20 | 0.5572 | 0.5000 | 76/80 |
| forward KL | 10 | 0.6563 | 0.5875 | 80/80 |
| forward KL | 20 | 0.5359 | 0.4875 | 75/80 |

Forward KL delayed the failure but did not turn it into a gain. These results
reject unconditional use of the zero-CoT distribution as a performance-improving
teacher on this pilot. The implementation remains useful as an explicit
experimental control; the CLI therefore requires `--teacher zero_cot` instead
of selecting it silently.

The next discriminating experiment should protect the frozen terminal policy
while testing whether the useful row-1 signal transfers: either mix a small
zero-CoT probability mass into the frozen terminal distribution, or gate the
early teacher using a predeclared, ground-truth-free confidence criterion. A
visual-attention snapshot remains a separate teacher family and should not be
presented as a successful fallback until its eager/full-attention extraction
and grid mapping are validated.

Raw local artifacts:

- `outputs/research_experiments/timeline_opd/opd_zero_cot_free_teacher_n5_d16_v1/`
- `outputs/research_experiments/timeline_opd/opd_zero_cot_reverse_n5_s20_d16_v1/`
- `outputs/research_experiments/timeline_opd/opd_zero_cot_forward_n5_s20_d16_v1/`
