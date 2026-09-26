# 时间线 bbox-only OPD：五样本真实运行记录

日期：2026-09-16。模型：Qwen3.5-0.8B。性质：探索性 effect pilot，非泛化结论。

## 当前结论

首轮没有复现“早期 teacher 总体定位更准”。已经实际进行了两轮各 20 步的 bbox-on-policy 蒸馏：reverse-KL 与 forward-KL 均降低平均 IoU。不能把代码跑通、次指标改善或个别图改善称为方法有效。

第二轮也已完成：首次坐标前锚点的 IoU 为 0.650169，完整去数字分支为 0.618404，均未超过匹配末端 0.665979。本轮总计完成两轮 teacher 对照（180 次出框）和两轮各 20 updates 的真实微训练。所有负结果保留；本轮不扩大样本或声称已经找到有效算法。

## 固定口径与样本

- 只用五张图：`row-0,row-1,row-3,row-4,row-5`；来自之前可完成执行的固定样本，不按本轮 IoU 筛选。这个既有选择仍限制总体代表性。
- 模型先自然生成完整 reasoning，再生成不带坐标的最终实体描述；所有条件共享同一条 reasoning 和实体。GT 只用于评分，不进入模型输入或 OPD 更新。
- reasoning 长度分别为 184、404、589、198、232 tokens。没有截断，但这些轨迹不足以检验真正的千 token 长链退化或分长度桶结论。
- 每条件、每图四个固定匹配 draw；四次抽样不是四个独立训练 seed，也不是四张独立图。主指标为五图等权的全输出平均 IoU；无效框计 0，不修复四角、不选择 best draw。副指标 Acc@0.5。
- bbox 统一 0–1000 整数坐标和同一输出 grammar，温度 1、top-p=1。`late_entity` 才是带实体 teacher 的匹配末端对照；`late_direct` 另行展示提示本身的影响。
- 微训练只训练末层 full-attention 的 query-only rank-8 LoRA，共 24,576 参数；reasoning/实体阶段关闭 adapter，整个主干冻结。

## 第一轮：teacher 自由出框

| 分支 | mean IoU | 对匹配 late_entity 的相对变化 | Acc@0.5 | 有效输出 |
|---|---:|---:|---:|---:|
| 完整推理直接出框 late_direct | 0.676300 | 非匹配提示对照 | 0.60 | 20/20 |
| 完整推理 + 最终实体 late_entity | 0.665979 | 基准 | 0.60 | 20/20 |
| Step-0 + 最终实体 early_step0 | 0.634820 | -4.68% | 0.70 | 20/20 |
| 首 span + 最终实体 early_span1 | 0.649357 | -2.50% | 0.75 | 20/20 |
| 全文四元组数字 scrub | 0.560302 | -15.87% | 0.55 | 19/20 |
| 等数量邻近字母 token sham | 0.665979 | 0.00% | 0.60 | 20/20 |

逐图原始聚合（每格为四次 draw 的平均 IoU）：

| 样本/目标 | reasoning tokens | late_entity | Step-0 | 首 span | tuple scrub | sham |
|---|---:|---:|---:|---:|---:|---:|
| row-0 / 被摩托车骑手载着的女人 | 184 | 0.348026 | 0.374920 | 0.436979 | 0.392764 | 0.348026 |
| row-1 / 最大的渔船 | 404 | 0.331606 | 0.509204 | 0.502599 | 0.331606 | 0.331606 |
| row-3 / 黄红公交车背面 | 589 | 0.857136 | 0.847473 | 0.847473 | 0.645609 | 0.857136 |
| row-4 / 白马 | 198 | 0.860112 | 0.688995 | 0.709874 | 0.660551 | 0.860112 |
| row-5 / 带蔬菜的披萨 | 232 | 0.933014 | 0.753509 | 0.749861 | 0.770977 | 0.933014 |

早期 teacher 改善 2/5 图、损害 3/5 图。因此 Acc@0.5 提升与平均 IoU 下降同时发生；不能事后把主指标换成 Acc@0.5。实体桥接文本经常只是重述原始 referring expression，未证明已经把图中实例歧义消除。

第一轮 scrub 只识别四数字元组，不识别 `x=850` 等具名坐标。row-1 因此没有被干预。第二轮扩展识别范围是新的协议条件，不覆盖或替换第一轮数据。

## 第二步：真的反向传播了两轮微训练

不是只生成 teacher 框：每次更新从当前 bbox 策略重新采样 token，冻结 teacher 在同一个学生坐标前缀和合法 support 上给出软分布，数字预测位置计算 KL 并反向传播。完整 reasoning/实体一次回放后固定；因此这是条件 bbox-on-policy 诊断，不是每次更新重新抽完整 reasoning 的大规模训练。

两轮均使用首 span teacher、20 updates、AdamW lr=0.001、rank=8、梯度裁剪 1.0。仅 OPD，`L_task=0`，没有 GT 或任务 reward；尚未验证完整的联合目标。训练/评测使用相同五图，评测 draw seed 与训练不同，不能称独立测试集。

| 损失 | step 0 IoU | step 10 IoU | step 20 IoU | 最终相对初始变化 | 最终 Acc@0.5 | 最终有效框 |
|---|---:|---:|---:|---:|---:|---:|
| reverse-KL p_student‖q_teacher | 0.651293 | 0.642431 | 0.580903 | -10.81% | 0.50 | 19/20 |
| forward-KL q_teacher‖p_student | 0.651293 | 0.645896 | 0.568146 | -12.77% | 0.50 | 19/20 |

两轮 step 0 的逐输出结果一致。teacher pilot 与微训练评测使用不同 draw seed，所以 0.665979 与 0.651293 不能被当作训练前后差异。

检查结果：各 20/20 updates 均有非零梯度；loss/gradient finite；adapter checkpoint 实际改变；冻结参数版本均未改变，未出现主干梯度。各轮最终一个无效框已计 0，没有剔除。CUDA 峰值 allocated 约 1.87 GiB（不含其他进程/驱动占用），不是完整训练集吞吐或部署延迟数据。

Flash Attention backward 提示非确定性；本轮配置为 `warn_only=True`。固定 seed 不等于已证明训练结果逐 bit 可复现。没有跨训练 seed 复现，故不报告显著性或 seed 均值/标准差。

## 第二轮 teacher 协议

不改五图、reasoning、最终实体、输出提示、draw seed 或评分口径，只增加如下分支：

1. `pre_coordinate`：选原轨迹首次被检测出的坐标数值之前最后一个自然 span 边界。无此前边界用 Step-0；没有坐标则用完整末端。只缩短 teacher 上下文，不缩短学生推理。
2. `scrub_all`：除四数字元组外也替换 `x/y/x1/x2/y1/y2` 具名赋值中的数值 digit tokens。保留变量下标和列表编号，替换为同 token 数的 `?`。
3. `sham_all`：替换同数量邻近非数字 token。此轮允许非数字标点，不同于第一轮仅字母 sham；应作为本轮配对控制解释，不能把跨轮差异全部归因于检测覆盖范围。

每图只做一次图像 prefill；各分支从原图 cache 出发逐 token 回放，避免当前 Transformers 的 GDN 多 token cached continuation 问题。没有每个 reasoning token 调重模型；scrub 回放有额外文本成本，不能宣称它等同免费 KV fork。

第二轮真实结果（60/60 输出有效，独立从 raw draws 重算）：

| 分支 | mean IoU | 相对 late_entity 变化 | Acc@0.5 |
|---|---:|---:|---:|
| 固定原末端 late_entity | 0.665979 | 基准 | 0.60 |
| pre_coordinate | 0.650169 | -2.37% | 0.75 |
| scrub_all | 0.618404 | -7.14% | 0.65 |
| sham_all | 0.531226 | -20.23% | 0.50 |

| 样本 | pre_coordinate token offset | 数值 tokens 被替换数 | pre_coordinate IoU | scrub_all IoU | sham_all IoU |
|---|---:|---:|---:|---:|---:|
| row-0 | 146 | 12 | 0.386534 | 0.392764 | 0.393015 |
| row-1 | 190 | 33 | 0.515592 | 0.507483 | 0.331553 |
| row-3 | 99 | 78 | 0.847473 | 0.700037 | 0.457420 |
| row-4 | 126 | 24 | 0.675074 | 0.660551 | 0.685533 |
| row-5 | 100 | 23 | 0.826174 | 0.831184 | 0.788606 |

扩大数字检测后，渔船一图的 scrub 从第一轮无干预的 0.331606 提升至 0.507483；然而全五图仍劣于原末端。scrub_all 优于 sham_all 也不能被称为净改善，因为 sham_all 本身比未干预末端显著差。这符合数字锚定值得继续研究的工作解释，但不能排除替换邻近标点/语义引起的不同破坏，不能据此证明文本注意力稀释机制。

`pre_coordinate` 相比首 span 对披萨较好、对白马较差，并没有统一解决 teacher 质量问题。因此没有继续扩大或额外训练这个尚无总体优势的 teacher；也没有依据 GT 为每图挑选最佳锚点。

## 可复现入口与原始证据

仓库：`/mnt/sda/sujingyang/research/routed-grounding-repair-verl`。git 基线 `4dacbccf68f8a9d318ef4815b53af12ba4a8d165`，含未提交研究源码；各 run manifest 另存入口 SHA256 与完整参数。

环境：`.venv/bin/python`，Python 3.12.14，torch 2.11.0+cu130，Transformers 5.5.3，xgrammar 0.2.2；复用 `.aris/compute/local.md` 中 `hf-live-cache@a093f958`。本轮 GPU 3。GDN 使用 torch fallback，耗时不代表优化内核性能。

源码：`timeline_self_distillation/run_teacher_pilot.py`、`run_opd_micro.py`、`run_teacher_refinement.py`、`terminal_adapter.py`。输出根目录 `outputs/research_experiments/timeline_opd/`：

- `teacher_pilot_n5_v1/`：120 条 bbox draws，完整 reasoning/实体、协议、manifest、summary。
- `opd_span1_reverse_n5_s20_v1/` 和 `opd_span1_forward_n5_s20_v1/`：各 20 条 train、60 条 eval、最终 adapter。
- `teacher_refinement_n5_v1/`：第二轮协议与运行结果。

训练命令（reverse/forward 各一次，已完成输出目录不可覆盖）：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 PYTHONHASHSEED=260600564 \
  .venv/bin/python -u -m timeline_self_distillation.run_opd_micro \
  --device 3 --teacher early_span1 --divergence reverse --steps 20 \
  --lr 0.001 --eval-draws 4 \
  --pilot-records outputs/research_experiments/timeline_opd/teacher_pilot_n5_v1/records.jsonl \
  --output-dir outputs/research_experiments/timeline_opd/opd_span1_reverse_n5_s20_v1
```

CPU 验证：`test_teacher_refinement.py`、`test_terminal_adapter.py`、`test_loss_reference.py` 共 19 passed。独立监控审查了第一轮 teacher 和两轮训练的原始记录；测试通过仅说明对应工程契约，不构成科研方法有效的证据。

第二轮独立审计亦已完成：60 draws、r/e、prompt、query、匹配 seed、mask 和原始指标全部核对一致。运行结束后 Ruff 仅整理了 `run_teacher_refinement.py` 的 import 顺序，所以当前入口文件 SHA 与 run manifest 中运行时 SHA 不同；实验逻辑未改变，19 项测试复验通过。

## 解释边界

1. 观察：早期分支总体 IoU 较低，两轮当前微训练退化。当前证据不支持“统一蒸馏最早状态就能改善总体几何”。
2. 工作解释：短上下文能修复部分错误，却也丢失部分有用定位信息；teacher 信号未先表现出总体优势。容量、学习率、20 步预算及纯 OPD 目标也是竞争解释，当前不能分离。
3. 已执行的下一判别：更晚但尚未写出数值的 teacher 仍无总体优势，故不直接扩大最早锚点训练。后续应先检验目标消歧/几何质量是否能在不看 GT 的情况下可靠区分，而不是用评测 GT 选出 best teacher；这是后续建议，本轮没有虚构这一能力已经实现。
4. 尚未实测：A 注意力快照蒸馏、独立测试集、真正长链长度分桶、SFT/PPO/GRPO/离线 KD 完整基线以及 task+OPD 联合训练。原方法文档是这些后续工作的设计，不是已完成结果。
