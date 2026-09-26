# 时间线 OPD 全面诊断：将实现、teacher、优化与目标现象分开

日期：2026-09-16。状态：代码/数据审计与三组补充 GPU 诊断完成。所有补充诊断仍限于原五图；没有新训练、没有覆盖旧结果、没有修改生产实现或另挑样本。主指标始终为全部输出 mean IoU，非法框计零；Acc@0.5 为次指标。

**结论：不能把这次失败概括成“teacher 太早”。最明确的新增证据是：模型确实学得更接近 teacher，GT-prefix 数字 NLL 也改善，但实际采样框的 IoU 下降。当前最值得优先检验的是 token 分布学习如何转化为稳定的几何决策，同时保留末端原本正确的信息。** 早期 teacher 的自由出框不稳定、纯 OPD 不含 task 项，以及更新幅度放大损害均有证据；各因素的独立因果贡献尚未分离。没有发现足以推翻原负结果的明确实现错误，也不能据五图推出一般 OPD 无效。

## 科学问题与可证伪假设

1. **teacher 质量不稳定**：早期分支可能修复错误图，却损伤正确图；需固定同一学生前缀评估分布，而不仅看 teacher 自由出框。
2. **实现错位**：若采样与带梯度回放不是相同策略，实际 BF16 多模态 logits/前缀 parity 应暴露差异；同 cache 的 self-teacher KL 应为零。
3. **优化或适配器不足**：固定前缀的 teacher KL 若未下降，不能说已经学会 teacher；若下降但 IoU 下降，则“完全没学到”不是充分解释。仍不能仅凭这一点排除容量或步长影响。
4. **token 与几何目标不一致**：KL 可以下降而 IoU 变差；数字序列的概率距离没有直接编码坐标相差一像素和一百像素的区别。
5. **目标现象未覆盖**：184–589 token 的自然轨迹并不是千 token 长链，不能凭这五图证实或证伪真正长链的空间退化机制。

## 已核实的旧实验事实

匹配 teacher 的完整推理末端 IoU 为 0.665979，Step-0 为 0.634820，首 span 为 0.649357，首次坐标前为 0.650169。早期分支改善 2/5 图、损害 3/5 图，并无统一早期优势。

原微训练使用了 **纯 OPD（L_task=0）**，不是原提案的 task+OPD 联合目标。仅有最后 full-attention 层 query-only rank-8 adapter，24,576 参数，20 updates；每图仅四次更新。所有 reasoning/实体固定，更新时仅 bbox 新采样。这个范围不能替代完整 OPD 算法的充分训练。

旧微训练逐图 mean IoU（相同评测 seed，各四次 draw）：

| 样本 | 初始 | reverse-KL 第20步 | forward-KL 第20步 |
|---|---:|---:|---:|
| row-0 女人 | 0.348168 | 0.313661 | 0.304552 |
| row-1 渔船 | 0.383049 | 0.342341 | 0.338674 |
| row-3 公交车 | 0.732120 | 0.727941 | 0.523650 |
| row-4 白马 | 0.860112 | 0.816842 | 0.860112 |
| row-5 披萨 | 0.933014 | 0.703730 | 0.813744 |
| 全五图 | 0.651293 | 0.580903 | 0.568146 |

这不仅是“原本正确的图被伤害”：在当前更新设置下，teacher 自由出框曾改善的前两张图，训练后也没有得到改善。因此仅说 teacher 太早仍不足以解释训练结果。

两轮各出现一个几何非法输出，原始字符串均为 `87,458,796,105]}</answer>`：格式完整、坐标在范围内，但 y_min > y_max。不是解析器把正确输出误判。其对总体降幅的配对贡献为 0.019146，占 reverse 总降幅的 27.20%、forward 的 23.03%；多数下降仍来自几何合法但定位变差的框。例如 forward 的一条公交车 draw 从 `[876,194,999,813]` 变成 `[983,0,1000,270]`，IoU 0.857136→0.013469。有效 bbox 只代表解析/值域/四角顺序合法，不代表 IoU 达标。

历史样本选择也不同：提案引用的早期优势来自**最终失败轨迹**，回答的是可挽救性；本轮五图含末端本来就较好的样本，必须同时看损害。不能用前者估计无条件蒸馏的总体收益。

## 数字锚定：真实证据与限度

在第一轮 `late_entity` 的 20 次输出中，16 次与 reasoning 中某个完整四元组逐坐标相同：女人、公交车、白马、披萨各 4/4。渔船使用具名坐标，没有被这个 tuple-only 统计覆盖，不能把它的 0/4 当成确定的“不复制”。

复制的数值可能错误，也可能接近 GT。`scrub_all` 0.618404 虽高于其 sham 0.531226，仍低于未干预末端 0.665979。替换数字不能被当作通用修复；这是一条值得进一步因果检验的线索，不是注意力稀释的证明。

对图像的定性检查也提示错误类型不同：女人图的末端框覆盖了更多骑手下半身，早期框更靠近被载女性的上身但仍不完整；白马图更多表现为框范围/上边界变化。该检查没有被用于筛样本或改 GT，也不是已完成的实体识别准确率评测。

## 新 GPU 对照一：只修改末端视觉读取

脚本 `timeline_self_distillation/run_visual_read_diagnostic.py`；原始输出 `outputs/research_experiments/timeline_opd/diagnosis_visual_read_n5_v1/`。

完整原 reasoning/实体和 prefix cache 保留；不训练、不调用早期 teacher。只在 bbox decode 时，对六个普通 attention 层的原始 image-key logits 加 0、+log(4)、-log(4)。所有条件都用相同显式 additive mask 和 SDPA math backend；避免将 backend 变化当成干预收益。图像每图编码一次，三个条件共享缓存。

| 条件 | IoU | 相对零偏置变化 | Acc@0.5 | 有效框 |
|---|---:|---:|---:|---:|
| 零偏置 | 0.651293 | 基准 | 0.55 | 20/20 |
| 图像 attention odds ×4 | 0.660166 | +1.36% | 0.60 | 20/20 |
| 图像 attention odds ÷4 | 0.652632 | +0.21% | 0.55 | 20/20 |

注意：这是对未归一化 attention 相对权重的倍率，不是归一化后的图像 attention 总质量恰好变成四倍。

零偏置与旧 SDPA 初始模型 **20/20 token IDs、bbox、IoU 一致**。两个干预各改变了 2/20 次输出；模型参数版本不变，全部输出有效，独立监控已核对 raw aggregation。

×4 的改善主要来自渔船（0.383049→0.426001），披萨仅有微小变化；÷4 在公交车有小幅变化。没有一致的正负方向响应，因此目前只支持“末端视觉读取存在局部可调空间”，不支持“末端退化完全由视觉注意力不足造成”。这也不是注意力快照 OPD 的训练结果。

## 新 GPU 对照二：固定前缀与 checkpoint 复验

脚本 `timeline_self_distillation/run_checkpoint_diagnostic.py`；输出 `outputs/research_experiments/timeline_opd/diagnosis_checkpoint_n5_v1/`。没有 optimizer，不做更新；加载两个现有最终 checkpoint，在同一批原 step0 的 20 条实际 bbox token IDs 上重新打分。teacher、base 和两个 checkpoint 使用完全相同的学生坐标前缀及合法 grammar support，仅在原协议的 digit-target predictor rows 上统计。

固定学生前缀结果（所有 numeric rows 合并取均值，nats；不是拿不同训练 rollout 的 loss 作比较）：

| 末端学生 | reverse KL：D(p‖q_early) | forward KL：D(q_early‖p) | 原固定评测 IoU |
|---|---:|---:|---:|
| 未训练 base | 1.272481 | 3.597677 | 0.651293 |
| reverse 训练后 | 1.107955 | 1.868678 | 0.580903 |
| forward 训练后 | 1.063420 | 1.497506 | 0.568146 |

两种 checkpoint 的两种 KL 都下降；五张图各自的 KL 也都下降。因此可以排除“完全没有向 teacher 学习”作为充分解释。改为图像等权聚合，reverse-KL 为 1.275112→1.111673/1.066903，forward-KL 为 3.604902→1.874122/1.501774，结论相同。

另一个意外但重要的结果：在**rounded GT 自身前缀**上打分，早期 teacher 和训练后模型的数字 NLL 都比 base 好。

| 条件 | GT-prefix digit NLL（token 合并） | GT-prefix digit NLL（图像等权） |
|---|---:|---:|
| base | 2.105246 | 2.122635 |
| early_span1 teacher | 1.521798 | 1.548275 |
| reverse checkpoint | 1.904650 | 1.926160 |
| forward checkpoint | 1.897796 | 1.919012 |

每张图单独也是这个方向。这说明“早期 teacher 完全没有有用概率信号”不符合该诊断。**但这些 NLL 只打分 GT 前缀上的数字行，既不是整条 bbox 的完整序列 NLL，也不是在错误学生前缀上续写的条件 IoU。** GT 前缀只用于打分，没有用于自由出框或任何参数更新；不能据此宣称 teacher 能纠正所有 on-policy 错误前缀。

综合来看，当前证据更支持：**token-level 概率学习和实际几何出框之间发生了脱节**。这与“teacher 自由出框质量不稳定”可以同时成立，不必二选一。数字 NLL 改善仍可能伴随采样分布扩散、前缀错误累积、坐标联合结构损坏或可学习参数化限制；本轮没有分别识别这些更细的原因，不能把其中任何一个写成已证实的唯一机制。

实现一致性检查已完成；主线程和独立执行监控均从 raw JSONL 重算，结果与 summary 一致：

- base、reverse、forward 各 20 条重采样，总计 **60/60 token IDs 和 IoU 与旧评测精确一致**。
- 同一个 late cache 的两条独立 fork 得到 self-teacher KL=0；zero-increment adapter 与 base 一致，两个 checkpoint 关闭 adapter 后恢复 base。
- 五份主 C0 各 48 个 cache tensor 的版本和内容 hash 不变；473 个冻结参数的版本不变。
- 真实 BF16 多模态中，五图 × 三个模型状态的 **15 个首 numeric row**，grad-enabled 与 no-grad 的合法 support log-prob 最大绝对差为 0，argmax 全一致。
- 最后一项只覆盖首数字行的 forward，**没有覆盖所有坐标位置的带梯度前向，也没有证明 backward/optimizer 数值等价**。不将局部 parity 包装为完整训练正确性的证明。
- 全部诊断无 NaN/Inf，无 optimizer 或参数更新；运行用时约 358 秒。

## 新 GPU 对照三：更新残差幅度，而非重新选学习率

脚本 `timeline_self_distillation/run_adapter_scale_diagnostic.py`；输出 `outputs/research_experiments/timeline_opd/diagnosis_adapter_scale_n5_v1/`。加载现有最终 checkpoint，A 不变，仅将 query-LoRA 的 B 乘以预先固定的比例。无优化步骤、无新 teacher，评测仍为同五图和原固定 seed。

| 残差比例 | reverse checkpoint IoU | forward checkpoint IoU |
|---|---:|---:|
| 0（关闭增量） | 0.651293 | 0.651293 |
| 1/4 | 0.644638 | 0.645840 |
| 1/2 | 0.613241 | 0.624351 |
| 1（原最终模型） | 0.580903 | 0.568146 |

两端均逐 token IDs/IoU 复现原 step0/step20。减少幅度可以减轻损害，但本次测试的两个非零缩放仍没有总体超过基线。它支持“较强更新放大了损害”，而不是“只要把现有更新缩小就已得到有效方法”。这是沿现有残差方向的只读干预，不等于以较低学习率重新训练；也没有穷尽所有强度、rank、层位置或优化器。

## 其他路线：优先解决哪里，而不是再换一个早期时刻

先区分原始 A/B 两路：本模型并非各层都带独立 visual cross-attention 的编码器—解码器结构，而是 18 个 Gated DeltaNet 层与 6 个 full-attention 层；可直接读取的 image-key attention 是后者 self-attention 中的视觉 token 子块。**A 的早期 attention 快照目前没有被验证为有效 bbox teacher**，中心/热图也不自动提供四角与范围。上面的视觉倍率实验只测试末端读取敏感性，不等于已经训练了 A。**B 的短分支确实已跑过**，自由出框不占优，但新的 GT-prefix NLL 留下了有用分布信号，不能简单判死刑。此外，当前冻结实体常只是复述 referring expression，尚未证明每例完成了实例消歧；这也是 A/B 都需要面对的共同问题。

1. **保留完整语义的独立视觉读出**：最终实体/state 查询一次编码后保留的视觉特征或 KV，bbox 专用 reader 输出几何分布。逻辑负责实体判定，几何分支保留直接访问图像的通道，不要求早期语言状态天然更准。主干仍冻结，但 reader 需要独立证明 grounding 能力；不能把 attention 中心直接当完整框。冻结 backbone 加可训练视觉查询模块有 [BLIP-2](https://proceedings.mlr.press/v202/li23q.html) 作为架构动机，不是本项目收益证据。
2. **有末端保留项的时序蒸馏**：在同一个学生坐标前缀上，将冻结 late 分布与少数前期分布做预先固定的概率混合，再蒸馏；或者用独立 dev 校准全局权重。这样不强迫覆盖 late 的全部正确知识。不能在这五图上按 GT 选择每图 best teacher；低熵也不等于正确。混合是逐 token 条件分布，不等同整条 sequence 的固定混合模型。
3. **改为数值/几何分布的训练目标**：保留 bbox-only loss，但在四坐标的数值分布或轻量 residual box head 上比较几何距离，而不只比较十进制 digit tokens。必须正确处理四角顺序、多峰性和坐标依赖；四个一维 Wasserstein 之和不是四维 bbox Wasserstein，也不直接等于 IoU。定位损失与 IoU 的错配有 [GIoU](https://arxiv.org/abs/1902.09630) 和 [分布式框表示](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html) 作为动机，未在本系统证明增益。

这三条仍需比较 bbox-only task 基线与 task+OPD，而不是只跑纯蒸馏。GT 只做诊断时必须与“额外监督训练”分开标注；不能用一小批样本上的 oracle selection 包装自蒸馏效果。

若继续投入训练，我的优先级是：**先用匹配的 bbox-only task 基线确认现有 adapter 能否改善几何输出，再单独加入保留 late 的蒸馏项；不要先扩大早期 teacher 的作用强度。** 如转向新结构，则优先考虑实体条件的静态视觉 reader，并使用几何感知目标；它不依赖“越早越准”的前提，但属于待验证新路线，不是本轮已经成功的修复。

一个保守时序分布可写作 `q_mix,t = (1-α) q_late,t + α q_early,t`，两者均在同一学生坐标前缀打分、detach，α 预先固定或由独立 dev 决定。只对 bbox 坐标 predictor rows 求 `L_task,bbox + λ D(q_mix,t, p_adapter,t)`；主干在 reasoning 阶段始终不启用 adapter。这里使用数值/几何距离时必须先定义值空间分布，不能把数字 token 的索引距离直接叫作坐标 Wasserstein。

关于 OPD：[GKD](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5be69a584901a26c521c2b51e40a4c20-Abstract-Conference.html) 支持在学生生成前缀上获取 teacher feedback，但不保证任何较早 teacher 都更好，也不使数字 mask + stop-gradient rollout 的 surrogate 自动成为精确 sequence-KL/IoU 梯度。关于机制：[Attention is not Explanation](https://aclanthology.org/N19-1357/) 是不能把 attention 图直接当因果解释的方法学提醒，其 NLP 结果不能直接外推为本 VLM 的反证。

## 排查边界与下一步判别

| 问题 | 本轮结论 | 尚未覆盖 |
|---|---|---|
| GT、坐标归一化、IoU、非法框计分 | 独立审计从 RefCOCOg/COCO 标注追溯并重算，通过 | 不等于其他数据集也通过 |
| loss 方向、token shift、grammar support、teacher detach | 与冻结微训练协议一致；4004 个四字段数值序列化检查无 digit/punctuation 混合 token | 当前统计是 digit-target 行，不是完整 bbox 序列 KL |
| cache fork / checkpoint /冻结范围 | 新 BF16 复验及 cache 内容检查通过 | 全位置 grad forward、backward parity 未完整覆盖 |
| teacher 太早导致全部失败 | 不支持这么简单的解释；自由出框不稳，但 GT-prefix NLL 更好 | 错误学生前缀下的条件续写 IoU、实例消歧质量尚未独立检验 |
| adapter 完全学不动 | 固定前缀 KL 五图都下降，不支持“完全没学到” | 容量、rank、层位置、训练步数仍未穷尽；无 matched bbox-task 训练基线 |
| 只需缩小现有更新 | 0.25/0.5 倍减轻伤害，但仍低于 base | 不等于低学习率重新训练；未扫描所有幅度 |
| 视觉注意力稀释 | 当前干预无一致正负方向响应，不能确认唯一机制 | 视觉信息也可能已融合进文本/GDN 状态，image-key 倍率不是删除全部视觉信息 |
| 真正 Long-CoT 退化被解决/否定 | 均不能声称 | 五条自然 reasoning 仅 184–589 tokens，无 ≥1024 token 桶或独立测试集 |

后续最高判别价值的两个诊断是：(i) 固定错误学生坐标前缀，比较 teacher 的条件续写几何质量；(ii) 在同一冻结 adapter 上区分 bbox-task 与 task+OPD。另可用预先固定的 greedy 与多 draw 对照区分模式偏移和采样散布，但不得事后更换本轮主指标或只保留有利 draw。这些是后续建议，**本轮没有声称已执行**。

独立完整审计的结论为 **WARN / same-family / provisional**：原始数值和数据来源通过；旧运行没有保存全部导入源码快照、refinement 的运行时源码 hash 与当前格式化后文件不一致，存在复现留档缺口。旧提案写“几何非法框不蒸馏”，实际微训练协议允许其数字行参与；这 40 个实际训练样本均几何合法，因此该差异未造成本次更新结果。保留这些问题，不用补充诊断抹去历史记录。完整审计覆盖旧四轮与视觉倍率诊断；后两组新诊断由主线程和独立执行监控核验，不冒称已纳入该份审计。

## 运行记录与复现入口

仓库：`/mnt/sda/sujingyang/research/routed-grounding-repair-verl`；revision `4dacbccf68f8a9d318ef4815b53af12ba4a8d165`，dirty 状态如各 manifest。模型：`/mnt/sda/sujingyang/models/Qwen3.5-0.8B`；torch `2.11.0+cu130`、Transformers `5.5.3`、xgrammar `0.2.2`；GPU 3；基 seed `260600564`。本轮直接使用 HF cache，不经过 vLLM，因此没有证据把当前蒸馏负结果归因于 vLLM 版本。

每个输出目录都有 `manifest.json`、`protocol.json`、原始记录及 `summary.json`；checkpoint 诊断另有 `parity.json`。manifest 记录完整调用参数、模型/输入路径、seed、checkpoint 和相关源码 hash；三次运行均不覆盖旧目录。

```bash
cd /mnt/sda/sujingyang/research/routed-grounding-repair-verl
export CUDA_VISIBLE_DEVICES=3
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=260600564
.venv/bin/python -u -m timeline_self_distillation.run_visual_read_diagnostic --device 3 --pilot-records outputs/research_experiments/timeline_opd/teacher_pilot_n5_v1/records.jsonl --reference-eval outputs/research_experiments/timeline_opd/opd_span1_reverse_n5_s20_v1/eval.jsonl --output-dir outputs/research_experiments/timeline_opd/diagnosis_visual_read_n5_v1
.venv/bin/python -u -m timeline_self_distillation.run_adapter_scale_diagnostic --device 3 --output-dir outputs/research_experiments/timeline_opd/diagnosis_adapter_scale_n5_v1
.venv/bin/python -u -m timeline_self_distillation.run_checkpoint_diagnostic --device 3 --pilot-records outputs/research_experiments/timeline_opd/teacher_pilot_n5_v1/records.jsonl --reverse-dir outputs/research_experiments/timeline_opd/opd_span1_reverse_n5_s20_v1 --forward-dir outputs/research_experiments/timeline_opd/opd_span1_forward_n5_s20_v1 --output-dir outputs/research_experiments/timeline_opd/diagnosis_checkpoint_n5_v1
```

以上为实际参数记录；如重跑须换成新的 output-dir，不能覆盖这次证据。本轮 CPU 回归检查五个 `timeline_self_distillation/test_*.py` 文件，**28 passed**，两条 Swig deprecation warnings 不影响判定。

- [原五图实验报告](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/docs/timeline_opd_five_sample_results_20260916.md)
- [独立完整审计](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/docs/timeline_opd_experiment_audit_20260916.md)
- [固定前缀诊断汇总](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/timeline_opd/diagnosis_checkpoint_n5_v1/summary.json)
- [真实多模态 parity 结果](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/timeline_opd/diagnosis_checkpoint_n5_v1/parity.json)
- [视觉读取对照](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/timeline_opd/diagnosis_visual_read_n5_v1/summary.json)
- [适配器缩放对照](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/outputs/research_experiments/timeline_opd/diagnosis_adapter_scale_n5_v1/summary.json)

## 后续补证：贪心与 16 次随机解码（同日完成，无训练）

上述“greedy 与多 draw 对照”后来已按用户明确指令完成，原文保留为当时的诊断记录。沿用相同五图、完整 reasoning、旧 entity、输出语法及温度 1；三个已有模型状态各做 5 个 greedy + 80 个 random 输出，合计 255 框。

| 状态 | 随机 mean IoU（主指标） | 非法框 / 80 | 贪心 mean IoU（仅诊断） |
|---|---:|---:|---:|
| Base | 0.667700 | 0 | 0.676242 |
| Reverse checkpoint | 0.574789 | 4 | 0.676242 |
| Forward checkpoint | 0.565851 | 3 | 0.676242 |

两个 checkpoint 的五个贪心 bbox 均逐坐标保持不变，但每图随机 mean IoU 均下降；支持优先调查非贪心备选的分布变化。不能仅由有限采样断言熵增加，也不能将该结果独立归因于 early teacher 或 entity bridge。主指标未换成 greedy，未降温、未训练、未混入新 bridge。60 条旧 seed 的 token IDs/bbox/IoU 全部精确复现，473 个主干参数版本未变；主线程和独立监控均已核验。此处新结果不是旧完整审计的追加覆盖。

[完整新报告与逐图数字分歧](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/docs/checkpoint_decode_diagnostic_20260916.md)；原始证据在 `outputs/research_experiments/timeline_opd/diagnosis_decode_n5_d16_v1/`。不能把新 16-draw base 的 0.667700 与旧 4-draw base 的 0.651293 当作模型变化。

## 后续补证：第二、第三段教师（同日完成，无训练）

用户随后要求测试 span2/span3。本轮固定旧 entity 和完整 reasoning，只改变累计 reasoning 前缀，四条件各 80 random + 5 greedy，共 340 框。随机 mean IoU：span1 **0.612680**、span2 **0.652442**、span3 **0.656017**、late **0.667700**。span2/3 相对 span1 提升 6.49%/7.07%，但主指标仍低于 late，逐图也不是单调改善（渔船 span3 比 span1/2 差）。所有框合法、85 条 late 精确复现、473 个参数和25份缓存检查均不变，独立核验通过。

支持将 span2/span3 保留为优于首段的候选，但不代表已用其完成 OPD 训练或修复 bridge。不同图的相同 span 编号语义进度不一致；白马第三段仍只是标题。详见 [教师 span1/2/3 逐图对照](/mnt/sda/sujingyang/research/routed-grounding-repair-verl/docs/teacher_span123_comparison_20260916.md)。
