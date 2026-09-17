# Natural reasoning prefix checkpoints

This directory is intentionally isolated from training, attention analysis,
visual-token mapping, probes, and bbox heatmaps. `run_pilot.py` preserves the
current Qwen3.5 Student prompt and two-phase decoding protocol, saves raw ids,
and slices reasoning only at exact token offsets.

New rollouts also write an artifact-closed replay bundle while the original
processor outputs are still in memory:

```text
OUTPUT_DIR/artifacts/SAMPLE_ID/
├── manifest.json
├── trajectory.json
├── checkpoints.jsonl
├── model_inputs.safetensors
└── original_image.png
```

`model_inputs.safetensors` stores the exact unpadded `prompt + natural
reasoning` inputs needed by the IVA replay: `input_ids`, `attention_mask`,
`mm_token_type_ids`, `pixel_values`, and `image_grid_thw`. The manifest records
the dtype, shape, and SHA-256 of every tensor plus the compact checkpoint
schedule. The complete token segmentation is retained in `trajectory.json`;
checkpoint prefixes remain direct slices of `reasoning_token_ids`.

vLLM does not return Qwen3.5 full-layer attention during generation. The bundle
therefore saves exact replay inputs rather than attention matrices. The IVA
probe loads these tensors directly and performs one teacher-forced Hugging Face
eager forward, without regenerating reasoning or rerunning the image processor.

```bash
PYTHONNOUSERSITE=1 .venv/bin/python reasoning_checkpoints/run_pilot.py \
  --phase all --device 2 --limit 50 \
  --output-dir reasoning_checkpoints/pilot_seed260600564_n50_rollout_artifact_v2
```

The reviewed pilot export is in
`reasoning_checkpoints/pilot_seed260600564_n50_boundary_v3/`:

- `trajectories.jsonl`: raw prompt, reasoning, close-tag, bbox-prefix, bbox-tail,
  response, and full-trajectory ids plus bbox/GT evidence.
- `checkpoints.jsonl`: one record per prefix checkpoint.
- `summary.json`, `checkpoint_texts.md`, `cases.md`: pilot inspection.
- `sanity_check.jsonl`, `sanity_check_report.{json,md}`: full-prefix and KV-cache
  numerical replay comparisons.

`token_offset` is the half-open count into `reasoning_token_ids`: a checkpoint
prefix is always exactly `reasoning_token_ids[:token_offset]`. The prompt-owned
`<think>\n`, generated terminal `</think>`, fixed bbox opener, and generated bbox
tail are stored separately. Text is decoded only for boundary observation and
human inspection; it is never encoded to reconstruct a reasoning prefix.

The boundary detector uses punctuation, newlines, and existing structural
markers. It defers marker-only list openers and ignores periods inside decimal
numbers or ellipses. These are checkpoint candidates, not claims that each
span is a true semantic reasoning step. A Segment Any Text hybrid was tested
on the same 50 trajectories and discarded because its small number of useful
extra boundaries did not justify a second tokenizer and threshold.

The v3 guard also keeps abbreviation/domain periods and punctuation inside
unclosed quotes, brackets, or inline code from becoming boundaries. Sentence
punctuation followed by closing quotes or Markdown markers is recorded at the
later exact token offset, after the structure closes.
