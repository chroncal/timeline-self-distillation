# MM-GCoT target bridge v6p1 提前停止证据收据

日期：2026-09-23

状态：**STOP**。v6p1 不进入几何诊断、lambda 校准或任何训练。

## 决策依据

`bridge_v6p1_dev70` 的冻结选择为 70 条（35 Attribute、35 Object），但实际只完成了 52/70 条 raw record。两个 partial review snapshot 共收录 49 条且无重复：

- `partial_review_snapshot_26.jsonl`：26 条（14 Attribute、12 Object）；
- `partial_review_snapshot_next23.jsonl`：23 条（12 Attribute、11 Object）。

已复审的 49 条中，按 target-entity bridge 契约可定位 8 条错误：3 条 Attribute、5 条 Object。Object 错误已经超过冻结容错：计划 Object 总数为 35，最多允许 3 条；复审到 23 条 Object 时已观察到 5 条，因此 `5 > 3`，即使剩余未复审 Object 全部正确，完整 35 条中的错误数仍至少为 5。

对应的下界为 `5/35 = 14.29%`，高于容错 `3/35 = 8.57%`。这是提前停止的充分依据，不需要把剩余样本的未知状态填成结果。

## 已定位的错误

错误口径沿用已有审核协议：`target_entity` 必须是支持最终定位的具体实体、部件或对象组；属性值不能代替实体；容器和内容、物体和其上/内的对象不能互换。以下只列入已复审且可由 frozen reasoning 与 raw record 直接核对的错误。

| 任务 | sample_id | raw `target_entity_reference` | 错误 |
|---|---|---|---|
| Attribute | `train:attribute:000002318109_cot:3295` | `The bedspread on the bed with the flowers` | reasoning 同时讨论上下两张床单，未唯一选定其中一张；该描述也没有消除歧义。 |
| Attribute | `train:attribute:000002321455_cot:8615` | `pizza` | 画面中有两个 pizza；裸的 `pizza` 不能唯一指向问题所问的 cheese 所在目标，丢失了题目中的区分关系。 |
| Attribute | `train:attribute:000002405012_cot:2739` | `rectangular` | 输出的是 shape 答案值，不是 clock 实体，违反 target/entity 与 task answer 分离的契约。 |
| Object | `train:object:000002363766_cot:16335` | `The man and woman are cutting the cake.` | target 字段是完整动作句，不是指向 man/woman 对象组的 noun phrase。 |
| Object | `dev:object:000002401077_cot:25705` | `The broccoli on the plate` | reasoning 最终选择白盘后方的椅子，reference 却指向盘中的西兰花。 |
| Object | `train:object:000002386060_cot:16571` | `The toasted sandwich on the white plate` | reasoning 讨论并回答 sandwich 内的 fillings，但 target 返回了外层 sandwich，发生 contents/container level 错位。 |
| Object | `train:object:000002403066_cot:22979` | `The laptop` | reasoning 最终识别的是 laptop lid 上的 grey phone，target 却返回包含它的 laptop。 |
| Object | `train:object:000002415264_cot:15061` | `the laptop` | reasoning 最终指向 laptop 上/屏幕关联的 computer monitor，target 却返回 laptop 本身。 |

五条 Object 错误分别可在复审快照中定位：`partial_review_snapshot_26.jsonl:19`、`:26`，以及 `partial_review_snapshot_next23.jsonl:6`、`:20`、`:22`。三条 Attribute 错误位于 `partial_review_snapshot_26.jsonl:7`、`:8`、`:15`。

这份收据不把学生本身选错、GT/问题语义的历史歧义、或尚未复审的样本擅自改写成 extractor 结果；已有 protocol 将这些轴分开记录。

## 复审边界

52 条已生成 record 的文件级核对结果为：

- `records/*.json`：52 个文件；任务分布为 27 Attribute、25 Object；
- `bridge_parse_status=valid`：52/52；
- `extractor_serialization_error=none`：52/52；
- `task_answer_status=usable` 与 `target_entity_reference_status=usable`：各 52/52；
- `reasoning_finish`：49 条 `stop`、3 条 `length`。

已生成但未进入两个 review snapshot 的 3 条为：

- `dev:object:000002327141_cot:24061`；
- `dev:object:000002410583_cot:7423`；
- `train:attribute:000002339011_cot:2293`。

这 3 条没有在本收据中赋予审核结论；另外 70−52=18 条尚无 v6p1 raw record。因此没有完成完整 70 条盲审，也没有完整 70 条的总错误率或通过率。

## 未开封集与后续闸门

`protocol.json` 明确记录 `gt_used_for_inference=false`、`student_choice_repair_allowed=false`、`sealed_independent_confirmation_accessed=false`。已有审核资料中的独立 48-image confirmation cohort 仍保持 sealed，未被打开或用于 v6p1 推断。

由于 Object 容错已在部分复审上被突破，v6p1 不得进入：

- L0/L/E/R 或其他几何诊断；
- lambda calibration；
- bbox-SFT、R-OPD、E-OPD 或任何正式训练。

本次工作只制作文件级证据收据，没有运行 GPU，也没有修改 raw records、protocol、review snapshots 或其他实验文件。

## 可追溯来源

- protocol：`outputs/research_experiments/mmgcot_timeline_training/bridge_v6p1_dev70/protocol.json`
  - `schema_version=mmgcot_target_bridge_v6p1_prompt_only`；
  - `samples=70`；
  - selection SHA-256：`5077091aff1aba0a1aa23bd6f9b9a62aea0f98d4007094f8a1463f33e81e89e9`；
  - protocol SHA-256：`e80629cf82acdad95a74ab0417cbcf9da0997c88de4c7c2e646a07f38c57e3c9`。
- raw records：`outputs/research_experiments/mmgcot_timeline_training/bridge_v6p1_dev70/records/`
- review snapshot 1：`outputs/research_experiments/mmgcot_timeline_training/bridge_v6p1_dev70/partial_review_snapshot_26.jsonl`
  - SHA-256：`99501203d307217021d5241524097ab4e33cb6ecde55eda55ba9483f76f132a1`。
- review snapshot 2：`outputs/research_experiments/mmgcot_timeline_training/bridge_v6p1_dev70/partial_review_snapshot_next23.jsonl`
  - SHA-256：`3ce37a7dc79e9756d90a8e4f2caf916d658f9e8459265ee1cff61ee8afc5207a`。
- error-label and gate contract：`docs/mmgcot_target_bridge_v2_protocol_20260922.md`。
- prior sealed-independent-set and no-geometry/training decision record：`docs/mmgcot_target_bridge_v3_prompt_only_results_20260922.md`。
