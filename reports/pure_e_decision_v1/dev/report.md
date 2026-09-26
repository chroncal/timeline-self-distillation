# Pure E-OPD A/B/C — dev

Primary metric: four random boxes → trajectory mean → image-equal mean IoU.
All arms use identical frozen trajectories and evaluation seeds; invalid outputs score zero.
Test-200 is a prior diagnostic cohort and is interpreted as a conditioned retest.

| Arm | Mean IoU | Training-seed SD | Acc@0.5 |
|---|---:|---:|---:|
| A: numeric-only terminal Q | 0.275707 | 0.000951 | 0.233333 |
| B: coordinate-decision terminal Q | 0.277589 | 0.001129 | 0.230556 |
| C: coordinate-decision last-two QVO | 0.274515 | 0.002958 | 0.227778 |

| Contrast | Paired mean IoU difference | 95% image-bootstrap interval |
|---|---:|---:|
| B-A | +0.001883 | [-0.000828, +0.005328] |
| C-B | -0.003074 | [-0.014781, +0.006946] |
| C-A | -0.001191 | [-0.013224, +0.009644] |

Bootstrap resamples images 10,000 times and retains all three training seeds within each sampled image. It does not substitute for the reported seed variation.

| Arm | Greedy mean IoU | Invalid random fraction | IoU≤0.05 random fraction | Median training step (s) | Peak memory (GiB) |
|---|---:|---:|---:|---:|---:|
| A | 0.349654 | 0.0139 | 0.3139 | 79.8 | 1.80 |
| B | 0.353705 | 0.0111 | 0.3042 | 77.7 | 1.80 |
| C | 0.341061 | 0.0097 | 0.3028 | 90.9 | 5.32 |

Coordinate digit-width counts are in `result.json` under `auxiliary`; training ending-decision KL is reported there for B/C.

Raw inputs and their SHA256:

- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/evaluation/dev/e_opd_pure_seed20260921.jsonl: `03845f23f56a429737f20a4af7300899ff5471633e3935e751d9944592ecf79a`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/evaluation/dev/e_opd_pure_seed20260922.jsonl: `ad0e3e550416c0d63fa5934da052304cdb3cf64402e8dfdcf56b2c6a872b521c`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/evaluation/dev/e_opd_pure_seed20260923.jsonl: `8eb2b451c4612220f38fa8747020d895938fc3811f920bdd32788a436352fa20`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/dev/terminal_q_seed20260921.jsonl: `c67dc5c7c1c5fbbc87d19a7fb7176b4402992e1401009473b89cf8488868adf7`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/dev/terminal_q_seed20260922.jsonl: `22d7893c4653001fcdc07f2f730e5862c41c278283b9cf3b63204faf3451c241`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/dev/terminal_q_seed20260923.jsonl: `4ccb5c2662c923e7189cf82973973cdee48f320192429d12cd0d21d8342aa03d`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/dev/last_two_qvo_seed20260921.jsonl: `407b189b442a45a62f3da74bb343de7d92d909af760329bd97455d3e34ffe6bb`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/dev/last_two_qvo_seed20260922.jsonl: `0ec70ae5799f0370bc246e9806d5acdccc983a8caf9e26ad01e63f9eb341afec`
- /mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/mmgcot_timeline_training/pure_e_decision_v1/evaluation/dev/last_two_qvo_seed20260923.jsonl: `fd84de1e456b16847420e6976a0db970fbb73790b9f7b99cea669a24f514fd09`
