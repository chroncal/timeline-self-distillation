# Timeline OPD 实验与实现审计

**总体：WARN；integrity_status：warn。**  
**review_independence：same-family；acceptance_status：provisional。**

原四组实验的负结果有真实数据支持。独立重算全部 **300 条原始评测 draw**，再核对新增视觉干预的 **60 条 draw**，未发现伪造 GT、自归一化指标、遗漏无效框或报告数值不符。没有发现已证实、足以使这批负结果失效的当前实现错误。

主要限制是：真实 BF16 多模态训练路径的 logits/梯度 parity 尚未直接验证；已运行 teacher 对照没有测设计中 H1 要求的“同学生前缀条件几何质量”；源码和运行状态的冻结记录不完整。因此负结果可以描述为“当前配置下未改善”，不能升级为“时间线蒸馏无效”或某一种失败机制已被证明。

以下路径均相对 `/mnt/sda/sujingyang/research/routed-grounding-repair-verl`。本审计只读，没有启动 GPU、修改文件或修复代码。

## A–F 检查

### A. GT 来源：PASS

验证链条为：

`refs(umd).p → COCO instances.json → manifest → train.parquet → trajectories.jsonl → teacher records → IoU`

逐一核对了五个样本的 expression、ref ID、annotation ID、image ID、原图尺寸、原图实际尺寸、原始 xywh、0–1000 xyxy 变换。全部一致。

| 样本 | image ID | annotation ID | ref ID |
|---|---:|---:|---:|
| row-0 | 529 | 205521 | 33507 |
| row-1 | 1166 | 178513 | 22086 |
| row-3 | 1762 | 169304 | 11180 |
| row-4 | 4535 | 54327 | 9951 |
| row-5 | 4642 | 1077114 | 7971 |

两个原始数据文件 SHA256 都与既有 metadata 完全一致：

- `instances.json`：`96c89b426c657f2f32247c16848ba67c3cb61fca74945da16c19e1027575c924`
- `refs(umd).p`：`0331c7533537b67c2f7ac8c8bab0da2d379d1754c6f5c110fd79f70e17e7bddb`

证据：

- `data/manifests/refcocog_umd_pilot/metadata/manifest_metadata.json:7`
- `data/manifests/refcocog_umd_pilot/metadata/source_hashes.json:5`
- `scripts/routed_grounding/build_manifest.py:560`、`:604`
- `scripts/routed_grounding/run_diagnostics.py:259`、`:276`
- `reasoning_checkpoints/run_pilot.py:252`、`:289`
- `timeline_self_distillation/run_teacher_pilot.py:284`、`:307`

GT 不进入 teacher query，`run_opd_micro.update()` 也不读取 GT。没有模型输出冒充 reference 的情况。

### B. 指标归一化：PASS

IoU 使用交并比；GT 归一化只依据原图宽高。不是按模型最大值、均值或最佳 draw 归一化。

- IoU：`verl/experimental/routed_grounding/router.py:134`
- 框合法性：同文件 `:82`
- teacher 等图权重聚合：`run_teacher_pilot.py:322`、`:345`
- 训练评测：`run_opd_micro.py:171`
- refinement：`run_teacher_refinement.py:344`、`:552`

三类无效输出均保留：

- 第一轮 `late_scrub`：row-3、draw 1，`[2,1000,999,828]`，角点顺序非法。
- reverse step 20：row-1、seed `269601564`，`[87,458,796,105]`。
- forward step 20：相同无效框。

全部 IoU=0，未修复角点或剔除样本。报告的相对变化同时附原始 IoU，未掩盖原始结果。

### C. 结果存在及精确数值：PASS 数值；WARN 可复现性记录

所有指定协议、manifest、summary、records、train、eval、eval_summary、adapter checkpoint 均存在。

独立从 token IDs 解码、重新 parse 和计算 GT IoU，确认：

| 条件 | mean IoU | Acc@0.5 | 有效框 |
|---|---:|---:|---:|
| late_direct | 0.676299797427103 | 0.60 | 20/20 |
| late_entity | 0.6659786549648222 | 0.60 | 20/20 |
| early_step0 | 0.6348199580699487 | 0.70 | 20/20 |
| early_span1 | 0.6493570817432004 | 0.75 | 20/20 |
| late_scrub | 0.5603015035599971 | 0.55 | 19/20 |
| late_sham | 0.6659786549648222 | 0.60 | 20/20 |
| pre_coordinate | 0.6501693428708852 | 0.75 | 20/20 |
| scrub_all | 0.6184037711303616 | 0.65 | 20/20 |
| sham_all | 0.5312255271364019 | 0.50 | 20/20 |

| 训练 | step 0 | step 10 | step 20 |
|---|---:|---:|---:|
| reverse KL | 0.6512925151604338 | 0.6424313702674929 | 0.5809029696287794 |
| forward KL | 0.6512925151604338 | 0.6458961936463185 | 0.5681463543463413 |

两轮 step 0 的完整 eval 记录一致，各有20次真实 update、60次 eval。40条 train 日志 loss/grad 均有限，grad_norm 均大于0。

两个 checkpoint 都仅含：

- `q_lora_down.weight`：`[8,1024]`，FP32；
- `q_lora_up.weight`：`[2048,8]`，FP32。

合计24,576参数。原本零初始化的 up 矩阵最终全部非零，证明参数确实更新：

- reverse up norm：`0.997696042060852`
- forward up norm：`1.2637430429458618`

证据：

- `docs/timeline_opd_five_sample_results_20260916.md:24`、`:53`、`:77`
- `outputs/research_experiments/timeline_opd/opd_span1_reverse_n5_s20_v1/eval_summary.jsonl:1`
- `outputs/research_experiments/timeline_opd/opd_span1_forward_n5_s20_v1/eval_summary.jsonl:1`

**可复现性缺口，P2，已确认：**

原四组 manifest 主要只 hash 入口脚本，关键被导入源码没有完整冻结；相关源码还是 untracked，Git revision 不能重建运行状态。refinement 的当前入口 SHA 与运行 manifest 不同：

- 运行记录：`a97e44e7852ea2aa7f07865d1de65260ec0b60a316642d487b0e235e3916ae3e`
- 当前文件：`36cee44b279be09256d40a01390c2f34942607b39717fca5c26b3af14be2e9fa`

报告说明是 import 排序，但仅凭现存 hash、没有运行时源码快照，不能独立证明“只变了 import”。这是来源固定不足，**不是数值造假证据**。

证据：`run_opd_micro.py:223`；refinement `manifest.json:2`、`:24`；结果文档 `:118`。

### D. 死指标路径：PASS 已报告指标；WARN 验证覆盖

已报告的 IoU、Acc、parse validity 确实从调用路径产生，并进入结果文件。

以下函数不被生产微训练调用：

- `loss_reference.py::masked_reverse_kl`
- `loss_reference.py::coordinate_wasserstein_1`
- `terminal_adapter.py::score_detached_cache_tokenwise`

生产训练在 `run_opd_micro.py:142` 手写 KL，并通过 `_advance` 推进。以上参考函数明确标为 CPU reference/witness；没有发现把未运行 W1 当实验结果报告的情况。

影响是：**reference 测试通过不能自动覆盖生产训练循环**。结果文档对此限定基本准确。

### E. 范围与选择：WARN，已披露

实际只有五张旧样本、一条保存的 reasoning/实体/图像，以及每种损失一个20步训练 run。四次 draw 是抽样重复，不是四个独立训练 seed。

`row-2` 在上游轨迹记录中是明确错误：

`ValueError: reasoning generation does not end with a single terminal </think>`

所以“此前执行成功样本”的描述可核实，未发现本轮按定位成功筛选。但完成性筛选仍限制代表性。reasoning 长度184/404/589/198/232，不能检验千 token 长链退化。

证据：

- `live_kv_probe_prototype/run_hf_fork.py:70`、`:124`
- `reasoning_checkpoints/run_pilot.py:324`
- `run_opd_micro.py:177`、`:253`
- 结果文档 `:13`、`:15`、`:49`、`:60`

当前文档将这些限制明确写出，没有泛化/显著性包装。

### F. 评测类型：PASS，分类如下

- teacher、refinement、训练评测：**real_gt**
- 新增视觉干预的 IoU：**real_gt**
- KL 训练监督：**self_supervised_proxy**，来自冻结模型，不是 GT 定位监督。
- cache/梯度/grammar witness：实现检查，不是定位效果评测。

应同时保留这两个层次：**无 GT 的训练监督，可以由真实 GT 评价其输出。**

## 实现检查

### 1. Token 对齐、mask 与 KL：当前路径未发现错位

`build_states()` 只消费 suffix[:-1]；采样、teacher scoring、student replay 都以 suffix[-1] 作为首输入，其 logits 预测第一枚坐标。后续消费上一枚学生 token，再预测当前 token。

证据：`run_opd_micro.py:58`、`:74`、`:116`、`:129`。

teacher 使用学生实际 token 前缀和学生该前缀对应的共同 grammar support；没有把 teacher 自己生成的另一条 bbox 按位置硬对齐。

我枚举了 **0–1000 × 四字段，共4004种 serializer**，当前 tokenizer 没有数字和标点混合 token；还重新用 grammar 验证了180条 teacher draw，全部完成。refinement 的 scrub/sham token ID 与 mask metadata 也从当前代码重算一致。

数值预测行的合法 support 除0–9外，可包含逗号或 `]`、`]}`、`]}</`。这符合设计的“数字预测位置上的全合法 support KL”，但不能描述成“只比较十个数字 token 的 KL”。

reverse/forward KL 方向、FP32 softmax、teacher detach、按数字位置数归一化都与各自声明一致。停止采样状态梯度及根据已采样 token 选 mask，意味着这是声明的 masked surrogate，不是完整序列 KL 的精确梯度。

### 2. 缓存与 GDN：历史错误已规避；真实完整 parity 未证明

当前 `_advance()` 无论收到多长 suffix，都逐 token forward，避免当前 Transformers 5.5.3 的多 token cached GDN 分支将 initial_state 设为 None。

- 项目：`live_kv_probe_prototype/run_hf_fork.py:216`
- 实际依赖：`.venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:433`、`:498`、`:510`

fork 克隆 conv/recurrent tensor；attention KV 共享，依赖 DynamicLayer 的 `torch.cat` 新建更新，与安装的实现一致：

- 项目 `run_hf_fork.py:160`
- 依赖 `transformers/cache_utils.py:119`、`:790`

微训练保存并恢复每图 rope_deltas；每分支位置由其自己的真实 cache 长度计算。未见将 early/late 强制使用相同位置的错误。

**P1 验证缺口，不是已确认 bug：**缺少实际0.8B/BF16/多模态路径上完整的 branch logits/state parity 与 rollout→gradient replay parity。现有21个CPU测试不能补上这一点。新视觉诊断的20/20生成 token parity 也不等同于训练 logits 或梯度 parity。

### 3. LoRA 范围：实现与限定一致

只冻结主干后安装末层 full-attention 的 query slice LoRA，按每个 head 的 `[query, gate]` 布局插入，未误把全投影前半当 query。K/V、gate、较低层不依赖 LoRA，因此缓存跨 token 的 detach 不会丢掉这份 surrogate 本应有的 adapter 路径。

证据：`terminal_adapter.py:58`、`:96`、`:118`；实际依赖 `modeling_qwen3_5.py:658`。

生产更新检查冻结参数无梯度，结束时检查 `_version` 未变。合理支持当前训练白名单；它不是完整模型 hash，也没有实际重生成训练后的 reasoning/e token parity。

### 4. CPU验证已重跑

- 原三文件：19 passed。
- 新视觉 mask 测试：2 passed。
- 均隐藏 CUDA、禁写 pycache/pytest cache；只出现非实质性 SWIG deprecation warnings。

### 5. 协议差异：存在，但未解释这40次更新

设计 `FINAL_PROPOSAL.md:154` 写“非法输出 OPD=0”；实际 `run_opd_micro.py:213` 明确声明几何非法框仍蒸馏数字行。这个实现与设计文本不一致，必须以运行协议解释。

不过40条保存的训练 bbox 都有效，故这项差异**没有证据表明影响了本次训练更新**。不应把它事后列为已确认退化原因。

## 科学解释方面的主要缺口

### P1：H1 尚未按其定义检验

设计 H1 是“同最终实体、**同学生 bbox 前缀**下，early teacher 条件几何质量更好”，并计划在四个坐标字段起点做条件补全。

- `FINAL_PROPOSAL.md:21`
- 同文件 `:294`

实际 teacher 对照均从空 bbox 前缀自由出框：`run_teacher_pilot.py:302`。生产训练虽计算同前缀 KL，但没有产生同前缀 teacher/late 的 GT 条件质量指标。

因此支持的是：

> 这些自由出框 teacher 在五图上的总体 IoU 没有超过 matched late。

不支持：

> 已直接证明训练所访问前缀上的 early 条件分布更差，或否定 H1 的所有形式。

### P2：负迁移原因仍不能分离

现有数据无法区分：

- teacher 在实际学生前缀上的质量不足；
- 最后一层 query-only 的可控范围不足；
- 学习率、20步预算、共享 adapter 的跨图干扰；
- 数字级 KL 与完整框 IoU 的目标差异；
- 纯 OPD 缺少 task objective；
- 未完成真实多模态训练 parity 的数值问题。

数据证明“更新后当前抽样评价变差”，不证明某一项原因。两种 divergence 都退化，只能排除“仅换这两个 divergence 就已经解决问题”。

### P2：scrub/sham 对照不能单独证明数字锚定

重算 mask 正确，但 `sham_all` 确实替换了括号、逗号、`x=` 等数值绑定结构及邻近语义；它不只是随机破坏等量无关文字。

证据：`run_teacher_refinement.py:215`；其 `records.jsonl:3` 的 row-3 sham 例如将 `[295,365,834,677]` 周围结构变为问号，且出现“the bus????876...”等文本。

`pre_coordinate` 同时去掉了后续语义、数字与上下文长度，亦非纯数字因果干预。原报告 `:89` 已正确保留这些替代解释。

## 新增 visual-read 诊断审计

**数值与限定：PASS；机制解释：仍须限定。**

三个条件都在同一个显式4D加法 mask、同一个 SDPA MATH backend 下运行；干预包装只包住 `sample_bbox()`，因此完整 r/e 与预构建 prefix 不受该干预重算。

证据：`run_visual_read_diagnostic.py:38`、`:52`、`:89`、`:146`。

独立重算60条结果：

| 条件 | mean IoU | Acc@0.5 |
|---|---:|---:|
| matched_zero | 0.6512925151604338 | 0.55 |
| visual_x4 | 0.6601658302567115 | 0.60 |
| visual_div4 | 0.6526323988967472 | 0.55 |

全部60条有效；matched_zero 与原step0的 **20/20 token序列完全一致**。新增 manifest 列出的5个 hash 均与当前文件吻合。

解释限制：

- `+log(4)` 是原图 key **相对于其他 key 的未归一化 attention odds** 乘4，不是保证最终 visual attention mass 乘4。
- x4只改变2/20条生成序列；/4也只改变2/20条，其中一条 bbox/IoU没有变化。
- x4与/4的总体IoU都略升，不构成清晰单调的“增加视觉读取就修复定位”关系。
- 干预六个full-attention层，比仅训练末层 query-LoRA 改变的范围大；也会影响 bbox 期间后续层及之后的缓存状态。因此不能据此直接定位为末层 query 容量问题。
- 20/20输出序列相同足以支持这些 seeds 下的 baseline 输出可比，不能称 bitwise logits/backend 或训练 gradient parity。

## 主张影响

| 主张 | 审计判断 |
|---|---|
| 已真实运行两轮teacher与两轮20步微训练 | supported |
| 当前五图上 early 自由出框没有总体IoU优势 | supported |
| 当前两轮微训练最终IoU下降 | supported |
| GT只评分，纯OPD更新不使用GT | supported |
| 只训练24,576个末层query-LoRA参数 | supported |
| 已证明H1条件教师假设错误 | unsupported |
| 已证明OPD不能改善定位 | unsupported |
| 已证明真实长链视觉稀释/数字锚定机制 | unsupported |
| 当前代码没有任何可能影响结果的实现问题 | unsupported；真实训练parity未完成 |
| 新视觉干预在当前固定draw上略提高平均IoU | supported |
| 新视觉干预证明视觉读取不足是主要原因 | unsupported |

## 紧凑 JSON

```json
{
  "audit_skill": "experiment-audit",
  "date": "2026-09-16",
  "overall_verdict": "WARN",
  "integrity_status": "warn",
  "review_independence": "same-family",
  "acceptance_status": "provisional",
  "read_only": true,
  "gpu_jobs_launched": 0,
  "checks": {
    "A_gt_provenance": {
      "status": "PASS",
      "details": "Five GT boxes traced to hash-verified RefCOCOg references and COCO annotations, with image dimensions and normalization verified."
    },
    "B_score_normalization": {
      "status": "PASS",
      "details": "Standard IoU and image-dimension normalization; invalid outputs retained at zero; no prediction-statistic normalization."
    },
    "C_result_existence": {
      "status": "WARN",
      "numeric_status": "PASS",
      "details": "All claimed artifacts and numbers verified; original runs lack complete dependency snapshots and refinement runtime entry hash differs from current file."
    },
    "D_dead_metrics": {
      "status": "PASS",
      "coverage_warning": "Reference losses and tokenwise scorer are not the production micro-training path."
    },
    "E_scope": {
      "status": "WARN",
      "images": 5,
      "training_runs_per_divergence": 1,
      "updates_per_run": 20,
      "draws_per_image_condition": 4,
      "reasoning_tokens": [184, 404, 589, 198, 232],
      "details": "Previously completion-successful seen images; no independent test set or training-seed replication."
    },
    "F_evaluation_type": {
      "status": "PASS",
      "bbox_evaluation": "real_gt",
      "opd_supervision": "self_supervised_proxy"
    }
  },
  "verification": {
    "original_raw_eval_draws_recomputed": 300,
    "visual_raw_eval_draws_recomputed": 60,
    "train_updates_checked": 40,
    "cpu_tests_passed": 21,
    "serializer_cases_checked": 4004,
    "mixed_digit_punctuation_tokens_found": 0,
    "teacher_grammar_draws_verified": 180,
    "visual_zero_backend_token_parity": "20/20"
  },
  "implementation_findings": [
    {
      "id": "I1",
      "severity": "P1",
      "kind": "validation_gap",
      "finding": "Actual 0.8B BF16 multimodal rollout-to-gradient replay and branch state/logit parity not directly established.",
      "confirmed_cause_of_negative_result": false,
      "evidence": [
        "timeline_self_distillation/run_opd_micro.py:129",
        "timeline_self_distillation/test_terminal_adapter.py:74",
        "refine-logs/timeline-opd-20260916/FINAL_PROPOSAL.md:262"
      ]
    },
    {
      "id": "I2",
      "severity": "P2",
      "kind": "reproducibility_gap",
      "finding": "Original run manifests do not freeze complete imported source; refinement runtime source unavailable at its recorded hash.",
      "confirmed_cause_of_negative_result": false
    },
    {
      "id": "I3",
      "severity": "P2",
      "kind": "documented_protocol_drift",
      "finding": "Actual protocol distills geometrically invalid bbox digit rows, while design says invalid outputs have zero OPD.",
      "observed_impact": "No invalid bbox among the 40 logged training updates."
    },
    {
      "id": "I4",
      "severity": "INFO",
      "kind": "verified",
      "finding": "Token shift, common grammar support, KL direction, teacher detach, last-layer per-head query slicing and sequential GDN continuation match the declared implementation."
    }
  ],
  "claim_impacts": [
    {"id": "C1_real_runs_and_reported_negative_iou", "impact": "supported"},
    {"id": "C2_early_free_generation_inferior_on_five_images", "impact": "supported"},
    {"id": "C3_H1_conditional_quality_rejected", "impact": "unsupported"},
    {"id": "C4_general_OPD_ineffectiveness", "impact": "unsupported"},
    {"id": "C5_numeric_anchoring_or_attention_dilution_proven", "impact": "unsupported"},
    {"id": "C6_visual_bias_fixed_draw_sensitivity", "impact": "supported_with_scope_qualifier"}
  ],
  "audited_primary_hashes": {
    "teacher_pilot_records": "27b07bc8c06e4faba58fd434e07683b2021a7740728bbe35d611e4aae85b4ef2",
    "teacher_refinement_records": "6808b513975f2550e7b9fe9bded5bfca0d647633452da5fca564fa5c9604bb9a",
    "reverse_eval": "ed26c9370fbe24ece5533764d5af8c9af71fe05dfa63a7bddd95e9e199abc3f5",
    "forward_eval": "029ae1f2122463884b649f638a7724f07adf97ca194587b4d88a7d4db53917d7",
    "visual_read_records": "a72a67f2562060b1397555e724d182dabf16f4c8e2714bcb1def5a0965dc3c0a",
    "reverse_adapter": "73b657dacbd1c1697b9047ed29e47b99491f51bb1feb94c172379a15449183ba",
    "forward_adapter": "c86935560157edbc0b3ec27b00413b8712321961999691b30ad17dfb629dfb6f"
  }
}
```
