# PROTOTYPE — live prefix-cache bbox probes

> 2026-09-16 audit correction: runs before the sequential-suffix fix passed
> multi-token suffixes into an existing Qwen3.5 cache. Transformers 5.5.3 GDN
> restarts its recurrent state on that code path. The recorded no-perturbation
> results for the main stream remain valid, but old probe bbox/IoU results are
> not evidence for a state-faithful early teacher. `_advance` now consumes
> cached suffixes one token at a time. Historical raw outputs are preserved;
> teacher-quality claims require a new run and branch-distribution validation.

Question: can short bbox probes be submitted while a natural reasoning request
remains live, reuse that request's cache, and leave the main reasoning token
sequence unchanged? This is deliberately a throwaway prototype. Two runners
are retained:

- `run.py` uses same-engine vLLM prefix-cache requests. It is **not** a public
  live-KV clone API, and Qwen3.5/GDN does not support vLLM batch invariance.
- `run_hf_fork.py` owns the Transformers decode loop and creates true
  copy-on-write cache forks. Full-attention prefix K/V tensors are shared;
  mutable GDN convolution/recurrent states are cloned.

## True-fork protocol

`run_hf_fork.py` tests one mechanism hypothesis: a bbox probe can run at a
causal reasoning boundary without recomputing the prefix or changing the
main rollout. The online execution order is:

1. Prefill the multimodal prompt and keep the main `DynamicCache`.
2. Sample the main reasoning continuation one token at a time with the fixed
   main seed. The runner keeps the generated prefix as token IDs.
3. Detect stable reasoning boundaries at sentence-ending punctuation or a
   newline, and also probe the terminal `</think>` boundary. The boundary
   prefix is an exact token-ID slice; it is never decoded and retokenized.
4. Fork the current main cache at that exact prefix.
5. On the fork, append either `</think><answer>{"bbox":[` or the natural
   current-best-box instruction followed by the same close/answer prefix.
   `xgrammar` constrains the remaining bbox tail.
6. Parse and record the fork's bbox result, then discard the branch.
7. Continue sampling from the untouched main cache until the main rollout
   finishes.

The fork is copy-on-write at the cache-component level. Full-attention
`keys`/`values` tensors for the existing prefix are shared: their layer update
creates new tensors, so the immutable prefix is safe to reuse. Qwen3.5 GDN
`conv_states` and `recurrent_states` are cloned because GDN decode mutates
those states in place. Cloning only the attention K/V, or shallow-copying the
entire cache without cloning GDN state, would let a probe contaminate the
main stream. The branch forwards only the close/instruction suffix and bbox
tail; it never forwards the reused main prefix.

The `baseline` condition runs the same Transformers decode loop and main seed
without creating forks. The two probe conditions use disjoint probe seeds.
The primary mechanism check compares each probed main rollout with its
baseline in raw token IDs, chosen-token log probabilities, and finish reason.

Frozen conditions:

- `baseline`: no probes;
- `forced_close`: append `</think><answer>{"bbox":[`;
- `natural_instruction`: append `Now stop and give your current best bounding
  box estimate.` before the same close/answer prefix.

The default five samples are the first five successful rows from the earlier
pilot: `row-0,row-1,row-3,row-4,row-5`.  Every condition uses the same explicit
main seed.  Probe requests use disjoint explicit seeds.  Reasoning prefixes are
always raw token-ID slices, never decoded and retokenized.

True-fork smoke (one sample):

```bash
RUN_TAG=$(date -u +%Y%m%dT%H%M%SZ)
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 PYTHONHASHSEED=260600564 \
  .venv/bin/python live_kv_probe_prototype/run_hf_fork.py --mode smoke \
  --output-dir outputs/research_experiments/live_kv_probe/hf_fork_smoke_${RUN_TAG}
```

True-fork frozen five-sample run:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 PYTHONHASHSEED=260600564 \
  .venv/bin/python live_kv_probe_prototype/run_hf_fork.py --mode formal \
  --output-dir outputs/research_experiments/live_kv_probe/hf_fork_formal_n5_v1
```

Each output directory is created once and never resumed or overwritten. The
script writes `protocol.json`, `manifest.json`, `records.jsonl`, and
`summary.json`. The frozen five-sample evidence and interpretation are in
[`outputs/research_experiments/live_kv_probe/hf_fork_formal_n5_v1/analysis.md`](../outputs/research_experiments/live_kv_probe/hf_fork_formal_n5_v1/analysis.md).
Console output surfaces the active condition, sample, checkpoint count, parse
status, and timing.

Render the saved five-sample probes as a step-through timeline. The image keeps
the ground-truth box visible while the slider moves through reasoning boundaries;
each step shows the newly completed reasoning span, the immediately generated
probe box, and its IoU:

```bash
.venv/bin/python live_kv_probe_prototype/render_timeline.py \
  --output /tmp/reasoning-bbox-timeline.html
```

## Parser validity versus localization correctness

`parse_valid` is a syntax/geometry check, not a semantic accuracy metric. A
parser-valid probe has the expected answer/tag shape, valid JSON with only a
four-number `bbox`, finite coordinates in `[0, 1000]`, and ordered positive
`xyxy` area. An IoU-correct probe additionally has `IoU >= 0.5` with the
source trajectory's ground-truth target box. A structurally valid box can
still localize the wrong object.

## Current five-sample evidence

The frozen formal run uses `row-0,row-1,row-3,row-4,row-5`, main seed
`260600564`, and one probe draw per boundary:

| sample | main tokens | boundaries | forced-close parser-valid | natural-instruction parser-valid |
|---|---:|---:|---:|---:|
| row-0 | 185 | 11 | 11/11 (100.0%) | 10/11 (90.9%) |
| row-1 | 405 | 38 | 27/38 (71.1%) | 35/38 (92.1%) |
| row-3 | 590 | 37 | 31/37 (83.8%) | 34/37 (91.9%) |
| row-4 | 199 | 11 | 10/11 (90.9%) | 10/11 (90.9%) |
| row-5 | 233 | 15 | 13/15 (86.7%) | 15/15 (100.0%) |
| all boundaries | — | 112 | 92/112 (82.1%) | 104/112 (92.9%) |
| sample-macro mean | — | — | 86.5% | 93.2% |

All 112 probes in each arm completed the constrained bbox grammar. Every
parser failure came from non-positive corner ordering; none came from
malformed JSON or tags. Localization was much lower: forced close hit the
target on 5/112 probes (4.46%), and natural instruction on 7/112 (6.25%).
Mean IoU over parser-valid boxes was 0.1629 and 0.1397, respectively. Thus
the parser-valid rates must not be reported as bbox accuracy.

The cache-fork mechanism passed its primary validity gate on these traces:
all 10 probed main rollouts exactly matched their no-probe baselines in token
IDs, chosen-token log probabilities, and finish reason. Each arm reused
56,780 main-prefix tokens without forwarding them on a branch. Branch forwards
consumed 2,594 tokens for forced close and 4,152 for natural instruction;
excluding cache-copy cost, the cache supplied 95.6% and 93.2% of the
prefix-plus-branch token work. Each fork shared about 6.23 MB of
full-attention prefix K/V and cloned 10.32 MB of mutable Qwen3.5 GDN state on
average.

These five traces support the no-perturbation/cache-reuse mechanism, not a
claim that either probe wording is an accurate localizer. Natural instruction
had a +6.7 percentage-point sample-macro parser-valid advantage, but the
paired sample-level one-sided Wilcoxon check was not significant (`p=0.1875`)
and the boundary-level McNemar result (`p=0.0169`) is descriptive because
boundaries within a sample are correlated. Prompt order should be
counterbalanced on a larger independent sample set before comparing wording.

## vLLM 0.29.0 limitation

`run.py` is a same-engine vLLM prefix-cache approximation. It does not expose
or emulate a public live-request KV-clone API, so its cache-hit observations
cannot establish the true fork protocol above. In the pinned vLLM 0.29.0
environment, initializing Qwen3.5-0.8B with `VLLM_BATCH_INVARIANT=1` fails
before the engine starts with:

```text
RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN.
```

The failure witness can be reproduced from the repository root with:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 PYTHONHASHSEED=260600564 \
  VLLM_BATCH_INVARIANT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /mnt/sda/sujingyang/research/envs/vllm-0.29.0-cu13/bin/python \
  live_kv_probe_prototype/check_vllm_batch_invariant.py
```

The captured log is
[`outputs/research_experiments/live_kv_probe/vllm029_gdn_batchinv_init/run_from_file.log`](../outputs/research_experiments/live_kv_probe/vllm029_gdn_batchinv_init/run_from_file.log).
Use `run_hf_fork.py` for the verified Qwen3.5/GDN copy-on-write probe; do not
present the vLLM approximation as a live KV clone.
