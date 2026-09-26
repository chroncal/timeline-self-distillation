# MM-GCoT 目标条件化时间线诊断

冻结 Qwen3.5-0.8B，回答早期分支是否定位更准、是否优于无推理历史的重新定位、是否能在学生的实际坐标前缀上提供更好的续写。运行不是训练，不产生模型检查点。

## 协议

正式协议由 `protocol.frozen_protocol()` 导出并保存在每次运行的 `protocol.json`。正式样本为 Attribute/Object Test 中各 100 张唯一图像；校验样本为 Trainval 中各 15 张，跨组图像不重复。所有样本由 seed 20260920 确定，参考 CoT、答案与 GT 不进入推理函数。

每题 3 条自然轨迹，temperature=0.8、top_p=0.95，上限 4096 tokens，无最低长度。自然完成后贪心提取目标描述；未解决或格式失败不使用人工/GT替代。E 为原轨迹前四分之一以内的最后句末，没有句末时使用精确四分之一 token 位置；L 保留完整推理，R 不保留推理，三者追加同一目标描述。L0 保留完整推理，只询问原问题的目标，不追加描述。四臂分别贪心一次和 temperature=1、全合法支持随机四次。

B 固定使用 L/random/draw0 第一个和第二个坐标后的原始 token 前缀。E/L/R 每个前缀续写四次，不按框质量挑选。记录完整语法支持分布、学生 token 概率与熵，不把 token 概率改善当成定位改善。非法框不修正、IoU计零。

思考被截断/EOS、实体未解决、缺失逗号前缀分别记录。报告同时列出预定全队列、可比较轨迹和盲审确认身份的子集分母。随机出框先平均 draw、再平均轨迹、最后图像等权；置信区间按图像配对 bootstrap 10000次。主指标保持 E-L，不依据结果增加筛选。

## 环境与入口

复用仓库 `.venv` 的 hf-live-cache 环境（torch 2.11.0+cu130、transformers 5.5.3、xgrammar 0.2.2）。GPU由命令明确指定，启动器检查空闲显存；不修改环境。现有 GDN torch fallback 是已知性能限制。所有带缓存续写均逐 token 推进，并保存/恢复图像的 rope_deltas。

```bash
PYTHONNOUSERSITE=1 .venv/bin/python -m pytest mmgcot_diagnostic -q
PYTHONNOUSERSITE=1 .venv/bin/python -m mmgcot_diagnostic.data --help
PYTHONNOUSERSITE=1 .venv/bin/python -m mmgcot_diagnostic.launch \
  --manifest /mnt/sda/sujingyang/research/datasets/mmgcot_20260920/pilot.jsonl \
  --output-dir outputs/research_experiments/mmgcot_timeline/pilot_v1 \
  --gpus 1,2,3,4,5
```

正式运行只在校验有效后启动，替换为 formal.jsonl 和新的 formal 输出目录。输出目录不可覆盖；源代码快照、权重与输入哈希、Git dirty 状态、完整命令和分片PID随运行保存。

## 文件与盲审

- `records/*.jsonl`：原始提示、轨迹、唯一目标描述、各臂框/续写、失败和完成记录。
- `distributions/*.jsonl.gz`：L的所有保存序列在E/L/R条件下的原始token前缀和完整合法支持分布。
- `workers/`：环境/权重/源代码收据、进度、异常及最终完成标志。
- `source_snapshot/`：运行时源码副本。
- 数据目录：源文件、样本清单、选择过程及排除原因，与运行结果分离。

实体盲审仅展示原图、问题、参考目标与提取描述，不展示条件名、模型预测框或IoU。独立两人核查，冲突由主代理裁决，不能确认则保持 uncertain。另有运行前固定50图的贪心框盲审，区分仍指向同一实体但边界漂移、实体切换、格式错误和不确定。没有完成盲审时，报告必须明示，不能宣称同实体几何退化或机制结论。
