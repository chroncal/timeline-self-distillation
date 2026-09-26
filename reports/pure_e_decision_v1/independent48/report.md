# Pure E-OPD A/B/C — independent48

Primary metric: four random boxes → trajectory mean → image-equal mean IoU.
All arms use identical frozen trajectories and evaluation seeds; invalid outputs score zero.
Test-200 is a prior diagnostic cohort and is interpreted as a conditioned retest.

| Arm | Mean IoU | Training-seed SD | Acc@0.5 |
|---|---:|---:|---:|
| A: numeric-only terminal Q | 0.359682 | 0.003341 | 0.326968 |
| B: coordinate-decision terminal Q | 0.359630 | 0.002521 | 0.329282 |
| C: coordinate-decision last-two QVO | 0.371304 | 0.004820 | 0.346065 |

| Contrast | Paired mean IoU difference | 95% image-bootstrap interval |
|---|---:|---:|
| B-A | -0.000052 | [-0.002758, +0.002903] |
| C-B | +0.011674 | [+0.002550, +0.020035] |
| C-A | +0.011622 | [+0.002105, +0.020502] |

Bootstrap resamples images 10,000 times and retains all three training seeds within each sampled image. It does not substitute for the reported seed variation.

| Arm | Greedy mean IoU | Invalid random fraction | IoU≤0.05 random fraction | Median training step (s) | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|
| A | 0.499999 | 0.0110 | 0.1921 | 79.8 | 1.80 |
| B | 0.499353 | 0.0104 | 0.1956 | 77.7 | 1.80 |
| C | 0.498319 | 0.0075 | 0.1800 | 90.9 | 5.32 |

Coordinate digit-width counts are in `result.json` under `auxiliary`; training ending-decision KL is reported there for B/C.

Raw inputs and their SHA256:

- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/evaluation/independent48/e_opd_pure_seed20260921.jsonl: `b70886eaa52c305a3ec3f4db02d76389ef6ee8f6f0792ca35fefd8368f429c68`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/evaluation/independent48/e_opd_pure_seed20260922.jsonl: `1089b859f6a9eb2ff15bef4ece90071354788626898891eb2362e9bc4d9310e3`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/evaluation/independent48/e_opd_pure_seed20260923.jsonl: `366a31348f629e86d75ace745421022435a1a3f5f9db732a5237e6f734a548ad`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/independent48/terminal_q_seed20260921.jsonl: `998039de9292004ea7e071cc20aba59dff4178d777ce061d47754510500c652b`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/independent48/terminal_q_seed20260922.jsonl: `a1752789b11035bf2e9f05330bf16937e1639acd6e78837e380d4802e436c7ed`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/independent48/terminal_q_seed20260923.jsonl: `2e72a7d79e8ce7c72186fcfb7a685443e1ddf74eefdf791d9178116e4bcb7727`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/independent48/last_two_qvo_seed20260921.jsonl: `2afab5bd2d5e4449827d9f19b9f1a7f8826aa8b29c6e6a825fbf2d2ca24cdf31`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/independent48/last_two_qvo_seed20260922.jsonl: `902bdfe819683b1d25983104f438224f061f111d599b8ae907e70e5b002323c7`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/independent48/last_two_qvo_seed20260923.jsonl: `ea4c143410028edc3f8d2f29936230a30b8c86d23508b10d0aae584e626bb2a1`
