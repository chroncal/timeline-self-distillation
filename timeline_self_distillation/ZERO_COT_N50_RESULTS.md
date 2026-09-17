# Zero-CoT teacher: 50-image results

Date: 2026-09-17 UTC

This is a 50-seen-image mechanism and optimization diagnostic, not a held-out
generalization result. The source artifact contained 55 attempted trajectories:
all 50 trajectories with `status=ok` were retained and the five execution
failures were excluded before looking at localization quality. No sample was
selected using IoU.

## Frozen protocol

- Model: Qwen3.5-0.8B.
- Teacher: untouched multimodal `C0 + BOX_OPEN`; no generated reasoning, entity
  bridge, or repeated question.
- Student: `C0 + complete saved reasoning_token_ids + BOX_OPEN`; no entity
  bridge or repeated question.
- The 50 saved reasoning paths contain 29,820 tokens. Re-rendered prompts and
  prompt token IDs were required to match the frozen artifact exactly.
- Four fixed bbox draws per image and condition (200 boxes); invalid boxes count
  as IoU zero. Draw seeds are bound to original `source_index`, not filtered row
  position.
- OPD: one reverse-KL update per image, 50 updates total, learning rate 0.001,
  rank-8 terminal bbox adapter. Ground truth is used only after generation for
  evaluation.

## Raw condition table

| condition | mean IoU | draw SD | Acc@0.5 | valid boxes |
|---|---:|---:|---:|---:|
| terminal Long-CoT student, step 0 | 0.605946 | 0.340262 | 0.640 | 194/200 |
| zero-CoT teacher | 0.634490 | 0.327289 | 0.690 | 197/200 |
| student after 50 reverse-KL updates | 0.496797 | 0.368366 | 0.515 | 187/200 |

Paired over the 50 per-image means:

| contrast | mean delta | image SD | approximate 95% t interval | wins / ties / losses |
|---|---:|---:|---:|---:|
| teacher minus initial student | +0.028544 | 0.319667 | [-0.062305, +0.119394] | 23 / 0 / 27 |
| final minus initial student | -0.109148 | 0.154357 | [-0.153017, -0.065280] | 4 / 16 / 30 |
| final minus teacher | -0.137693 | 0.280662 | [-0.217457, -0.057929] | 13 / 0 / 37 |

The teacher's aggregate advantage is therefore heterogeneous and uncertain,
not a 50-image-wide improvement. Two samples have teacher gains above 0.91 IoU,
while several samples have large negative deltas. In contrast, the OPD damage is
broad and its paired interval is entirely below zero.

## Relation to reasoning length

Teacher-minus-student IoU delta has no detectable monotonic association with
saved reasoning length: Pearson r=-0.0608 (p=0.675) and Spearman rho=0.0716
(p=0.621). Length-bin diagnostics are also non-monotonic:

| reasoning length bin | token range | images | teacher delta | final OPD delta |
|---|---:|---:|---:|---:|
| shortest | 70-255 | 12 | -0.06155 | -0.05685 |
| lower-middle | 265-363 | 12 | +0.10332 | -0.15493 |
| upper-middle | 370-698 | 12 | +0.12768 | -0.09386 |
| longest | 744-1934 | 14 | -0.04330 | -0.12785 |

This run does not support a simple claim that longer reasoning monotonically
causes larger spatial degradation. It supports a weaker statement: early and
late spatial policies are heterogeneous and sometimes complementary.

## Interpretation and next discriminating test

The implementation works end to end, and the teacher can be better on average,
yet numeric-token KL does not transfer that average advantage. A likely
explanation is that one global adapter update changes bbox alternatives for all
samples while teacher quality varies by instance; pure OPD also contains no
frozen-terminal preservation term. This is an interpretation, not a proven
mechanism.

The next controlled test should preserve the frozen terminal policy and add
only a small zero-CoT contribution, for example an 80% terminal / 20% zero-CoT
probability mixture scored on the same student prefixes. A ground-truth-free,
predeclared confidence gate is the other candidate. Either must be compared
against the frozen-terminal-only control before attention snapshots are treated
as a replacement teacher family.

Raw local artifact:

`outputs/research_experiments/timeline_opd/opd_zero_cot_terminal_direct_n50_s50_d4_reverse_v2/`

The earlier `..._v1` directory is an intentionally preserved interrupted run
with no reported metrics; it was stopped during cache construction to correct
seed stability and frozen-prompt validation before this run.
