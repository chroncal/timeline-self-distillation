# Teacher conditioning contracts

## Experimental teacher: `zero_cot`

`run_opd_micro.py` now defines the teacher at `C0`, immediately after the
multimodal chat template opens the assistant thinking block:

```text
C0 = image + original user instruction + assistant "<think>\n"
teacher = fork(C0) + '</think><answer>{"bbox":['
```

The teacher consumes no generated reasoning token. It also receives no entity
bridge and no repeated copy of the question. The student path is unchanged: it
uses the complete saved long reasoning rollout and the terminal entity query.
Only bbox-token probabilities are distilled.

This is a **question-conditioned zero-CoT teacher**, not text-free pure visual
perception: the original instruction is already part of `C0`. The runtime fails
closed if the rendered prompt is not at the empty assistant thinking state or
if the fixed bbox suffix does not round-trip exactly through the tokenizer.
Each run records the resolved conditioning contract in `protocol.json`.

Example:

```bash
PYTHONHASHSEED=260600564 python -m timeline_self_distillation.run_opd_micro \
  --pilot-records /path/to/records.jsonl \
  --output-dir /path/to/output \
  --teacher zero_cot
```

`--teacher zero_cot` is required explicitly. It is intentionally not a default:
the first real five-image, 16-draw pilot found that this teacher was
geometrically mixed and that unconditional OPD reduced mean IoU. See
`ZERO_COT_PILOT_RESULTS.md`. A later 50-image run found a small, heterogeneous
teacher advantage but again found large negative transfer from unconditional
OPD; see `ZERO_COT_N50_RESULTS.md`.

## Legacy controls

The older choices remain available only to reproduce and diagnose historical
experiments:

- `early_step0`: `C0 + QUERY(entity, original question) + BOX_OPEN`
- `early_span1`: `C0 + first reasoning span + QUERY(...) + BOX_OPEN`
- `late_scrub`: `C0 + scrubbed full reasoning + QUERY(...) + BOX_OPEN`
- `late_entity`: `C0 + full reasoning + QUERY(...) + BOX_OPEN`

These are entity-conditioned controls and must not be reported as pure
perception anchors.

## Attention-map teacher is a separate experiment

An internal visual-attention snapshot is not interchangeable with a bbox-token
teacher. The present training path uses SDPA and cached decoding; the existing
attention diagnostic obtains maps by eager, no-cache trajectory replay and only
has conventional maps for full-attention layers. Converting those maps to a
visual-token grid and defining a loss therefore requires a separate, explicitly
validated experiment. It is not silently substituted for `zero_cot` in this
fix.
