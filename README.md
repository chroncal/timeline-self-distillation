# Timeline Self-Distillation：MM-GCoT 研究代码备份

2026-09-26 的服务器工作区快照。当前主线是 **纯 E-OPD 的 A/B/C 对照**：在相同末端定位条件下，分别检验坐标结束决策监督和更大的 bbox 阶段适配器是否改善定位。

实验源码按原样复制，包括源仓库未提交的研究文件；没有为了备份重写 prompt、loss、模型接口或历史实验。文件来源及 SHA-256 见 `BACKUP_MANIFEST.json`，整个备份的校验清单见 `SHA256SUMS`。此前 2026-09-17 的备份仍保留在 Git 历史中。

## 当前方法

1. 冻结 Qwen3.5-0.8B，生成自然推理，再用固定 **v3p5** 提取器得到任务答案和目标实体描述；仅目标实体描述进入定位提示。
2. 学生 L 使用完整推理；教师 E 使用同一轨迹前约 25% 的推理前缀。两者共享原图、问题和晚期目标描述。
3. 当前学生每次重新采样 bbox；冻结教师沿学生的同一串原始坐标 token 前缀评分。
4. 在完整合法 grammar support 上计算 `KL(student || teacher)`。当前 A/B/C **没有 SFT、GT 坐标训练损失或 GRPO**。
5. 适配器仅在 bbox 输出阶段开启，自然推理和目标提取不更新。

| 配置 | 监督位置 | 可训练适配器 | 参数量 |
|---|---|---|---:|
| A | 实际生成数字的预测位置 | 第 23 层 Query，rank 8 | 24,576 |
| B | 数字及可选择的坐标结束位置 | 同 A | 24,576 |
| C | 同 B | 第 19、23 层 Query / Value / Output，rank 8 | 122,880 |

三组独立从原模型初始化；C 不是从 B checkpoint 继续训练。Query/gate 的交错布局得到显式处理，gate 不更新。C 的 bbox 内跨 token 反传使用本地可微 GDN 状态实现，未修改已安装的 Transformers。

冻结设置：学习率 `1e-4`，OPD 系数 `0.1`，有效 batch 16，microbatch 1，每个 run 200 steps，seeds `20260921/20260922/20260923`，固定评价 step 200。详见 [configs/mmgcot_pure_e_decision_v1.json](configs/mmgcot_pure_e_decision_v1.json)。

## 代码入口

| 路径 | 用途 |
|---|---|
| `mmgcot_timeline_training/formal_pure_e_decision.py` | 当前 B/C 训练、评估、真实模型验收；也提供 A 兼容模式 |
| `mmgcot_timeline_training/coordinate_decision.py` | 依据生成前 grammar 状态决定监督位置 |
| `mmgcot_timeline_training/expanded_bbox_adapter.py` | 两层 Q/V/O 适配器 |
| `mmgcot_timeline_training/functional_gdn.py` | 保留 bbox 内跨 token 梯度的状态推进 |
| `mmgcot_timeline_training/formal_pure_opd.py` | 旧纯 R/E-OPD，其中纯 E 是 A 基线 |
| `mmgcot_timeline_training/formal_train_v2.py` | 共用模型/缓存引擎，以及历史 SFT+OPD 实验 |
| `mmgcot_timeline_training/prepare_v2.py` | 冻结自然轨迹与 v3p5 描述 |
| `mmgcot_timeline_training/formal_contexts.py` | L/R/E 条件和首位坐标预测边界 |
| `mmgcot_timeline_training/bridge_v3.py` | 当前 v3p5 prompt、grammar、parser；其他 bridge 文件是历史试验 |
| `mmgcot_timeline_training/e_decision_report.py` | 图像等权指标、配对 bootstrap 和结果图 |
| `scripts/run_e_decision_smoke.sh` | A 兼容性、B/C 短跑和真实 BF16 验收 |
| `scripts/run_e_decision_matrix.py` | 六个 B/C 正式 run 及共同轨迹评估的原调度脚本 |
| `mmgcot_diagnostic/` | 数据适配、L/R/E/L0 诊断及学生坐标前缀续写 |
| `timeline_self_distillation/` | 末层 Query 适配器、参考 loss 和早期诊断代码 |
| `reasoning_checkpoints/`, `live_kv_probe_prototype/` | 原始 token 轨迹、缓存分支和逐 token 续写工具 |
| `scripts/routed_grounding/`, `verl/` | 上述工具实际导入的历史依赖源码，保持原导入结构 |

`verl/` 是依赖快照，其存在不表示当前纯 E-OPD 使用 VERL PPO/RL 训练器。原 Apache-2.0 LICENSE 和源码版权声明保留。

## 实际数据规模与结果

这轮是 **310 图的小规模训练对照**，不是 MM-GCoT 全量训练。源训练文件有 63,528 条记录、4,999 个不同图像路径；本轮候选匹配到的本地原图有限，最终冻结 train 310、dev 60、独立测试 48、历史 Test-200。每图一题，Attribute/Object 两类。

- train：参与更新；自然轨迹每图固定 1 条，bbox 每次访问重新采样。
- dev：此前用于选择超参数；保存辅助结果，不作为额外的效果通过门槛。
- 独立 48 图：预先指定的主要测试集，每图 3 条共同轨迹，每条 1 次贪心和 4 次随机出框。
- Test-200：此前参与诊断和方案选择，属于既有诊断集复测。

主指标为随机框先在轨迹内、再在图像内平均，最后图像等权的 mean IoU。`Acc@0.5` 使用 `IoU > 0.5`。非法框计零，不修复角点、不挑选样本或 checkpoint。

本次备份包含六个新训练 run 完成后的 dev 与独立 48 图汇总；尚不包含 Test-200 完整结果。独立 48 图三个 seed 的均值：

| 系统 | mean IoU | Acc@0.5 |
|---|---:|---:|
| A | 0.359682 | 32.6968% |
| B | 0.359630 | 32.9282% |
| C | 0.371304 | 34.6065% |

B−A 为 −0.000052，图像配对 95% 区间 [−0.002758, +0.002903]；C−B 为 +0.011674，区间 [+0.002550, +0.020035]。结果支持该设置下扩大 bbox 适配范围改善随机定位质量，不直接证明注意力稀释或视觉证据使用机制。

完整汇总、每 seed 数字、辅助指标和图在 [reports/pure_e_decision_v1](reports/pure_e_decision_v1)。dev 的结果完整保留。报告中的原始记录路径指向原服务器，不表示原始 JSONL 已上传。

## 恢复与运行

这是保持现场实现的源码备份。**权重、Visual Genome 原图、数据标注、冻结轨迹、prefix cache、原始 rollout 和 adapter checkpoint 未上传。** 已有 `reasoning_checkpoints/` 历史符号链接原样保留，目标输出目录不在备份内。

实际 Python/依赖版本在 `environment/runtime.json`、`environment/packages.json`；源项目的环境文件在 `environment/original_project/`。核心版本包括 PyTorch `2.11.0+cu130`、Transformers `5.5.3`、xgrammar `0.2.2`。原环境文件包含历史可选后端，不是最小安装清单。

优先使用原服务器已验证的环境。换机器时，先准备兼容环境及外部工件，调整 `configs/mmgcot_formal_v2.json` 的数据清单路径、命令中的模型/缓存路径，以及 shell 调度脚本中的服务器目录。外部冻结清单的哈希仍必须匹配；改用新数据应创建新实验配置和输出目录。

从仓库根目录执行。B 的训练入口如下，路径为占位符，输出必须使用新目录：

```bash
python -m mmgcot_timeline_training.formal_pure_e_decision \
  --mode train --supervision-scope coordinate_decision \
  --adapter-scope terminal_q --device 0 --seed 20260921 \
  --model /path/to/Qwen3.5-0.8B \
  --selection /path/to/train_frozen.jsonl \
  --prepared /path/to/prepared/train \
  --cache-root /path/to/prepared/train/prefix_cache \
  --output /path/to/new_run --steps 200 --batch 16 \
  --lr 1e-4 --lambda-opd 0.1 --checkpoint-every 50
```

C 将 `--adapter-scope` 换为 `last_two_qvo`。A 兼容模式使用 `--supervision-scope numeric --adapter-scope terminal_q`。长期任务放入 tmux；原调度脚本固定了 GPU 分配，运行前需核对占用。

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  tests/mmgcot_timeline_training tests/test_formal*.py \
  mmgcot_diagnostic/test_*.py \
  timeline_self_distillation/test_*.py reasoning_checkpoints/test_*.py

sha256sum -c SHA256SUMS
```

本次备份验证记录见 `BACKUP_VALIDATION.json`。已在服务器原环境执行备份验证，未声称在新环境完整重跑训练。CPU 单元测试不代替新机器上的真实 BF16、缓存和 GPU 验收；所需验收入口已备份。
