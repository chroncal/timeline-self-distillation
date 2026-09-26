"""Frozen HF inference for the preregistered MM-GCoT timeline diagnostic.

Run separate deterministic sample shards on independent GPUs. All generated
IDs are journalled before any bbox branch; exact-prefix probability records
are stored separately so aggregation does not need to load them.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from mmgcot_diagnostic.protocol import (
    BOX_OPEN, BOX_REGEX, ENTITY_PROMPT, ENTITY_REGEX, MODEL, SEED,
    bbox_suffix, comma_prefix, early_offset, file_hash, frozen_protocol,
    iou, model_input, parse_box, question_prompt, stable_seed,
)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def append(path, value):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def cache_fingerprint(cache):
    return [(i, name, list(t.shape), tensor_hash(t))
            for i, layer in enumerate(cache.layers)
            for name in ("keys", "values", "conv_states", "recurrent_states")
            if isinstance((t := getattr(layer, name, None)), torch.Tensor)]


def sentence_offsets(tokenizer, ids):
    # Reuse the protected-span/decimal/abbreviation handling, but reject the
    # newline-only boundaries accepted by the old reasoning-span extractor.
    from reasoning_checkpoints.extractor import natural_boundary_offsets
    result = []
    for end in natural_boundary_offsets(tokenizer, ids):
        text = tokenizer.decode(ids[:end], skip_special_tokens=False).rstrip()
        if re.search(r'[.!?。！？][\"\u201d\u2019\)\]]*$', text):
            result.append(end)
    return result


class Engine:
    def __init__(self, model_path):
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        from live_kv_probe_prototype.run_hf_fork import _advance, _fork_cache_cow, _sample_top_p
        self.advance = _advance
        self.fork = lambda cache: _fork_cache_cow(cache)[0]
        self.sample = _sample_top_p
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.image_processor.size = {"shortest_edge": 3136, "longest_edge": 262144}
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda").eval()
        self.model.requires_grad_(False)
        self.parameter_versions = [p._version for p in self.model.parameters()]
        self.compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.model.config.text_config.vocab_size))
        self.box_grammar = self.compiler.compile_regex(BOX_REGEX)
        self.entity_grammar = self.compiler.compile_regex(ENTITY_REGEX)
        close = self.tokenizer.encode("</think>", add_special_tokens=False)
        if len(close) != 1:
            raise RuntimeError("thinking close must have one token")
        self.close_id = close[0]

    def ids(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.tokenizer.decode(ids, skip_special_tokens=False) != text:
            raise RuntimeError("suffix does not round-trip exactly")
        return ids

    def prefill(self, inputs):
        from PIL import Image
        with Image.open(inputs["image_path"]) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"},
                    {"type": "text", "text": question_prompt(inputs["question"])}]}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        if not rendered.endswith("<think>\n"):
            raise RuntimeError("native thinking template missing")
        data = self.processor(text=[rendered], images=[image], return_tensors="pt",
                              min_pixels=3136, max_pixels=262144)
        prompt_ids = data["input_ids"][0].tolist()
        out = self.model(**{k:v.to("cuda") if isinstance(v, torch.Tensor) else v for k,v in data.items()},
                         use_cache=True, logits_to_keep=1, return_dict=True)
        self.rope_deltas = self.model.model.rope_deltas.detach().clone()
        return out.past_key_values, out.logits[:, -1, :], prompt_ids, rendered

    def restore_rope(self):
        self.model.model.rope_deltas = self.rope_deltas.clone()

    def replay(self, c0, reasoning_ids):
        self.restore_rope()
        cache = self.fork(c0)
        if reasoning_ids:
            cache, logits = self.advance(self.model, cache, reasoning_ids)
        else:
            logits = None
        return cache, logits

    def rollout(self, c0, first_logits, seed, limit=4096, progress=None):
        self.restore_rope()
        cache, logits = self.fork(c0), first_logits
        rng = torch.Generator(device="cuda").manual_seed(seed)
        ids, logps, finish = [], [], "length"
        for _ in range(limit):
            if not torch.isfinite(logits).all():
                raise FloatingPointError("nonfinite reasoning logits")
            token, lp = self.sample(logits, generator=rng, temperature=0.8, top_p=0.95)
            if token == self.close_id:
                finish = "stop"
                break
            if token == self.tokenizer.eos_token_id:
                finish = "eos"
                break
            ids.append(token)
            logps.append(lp)
            cache, logits = self.advance(self.model, cache, [token])
            if progress is not None and len(ids) % 128 == 0:
                progress(len(ids))
        return cache, logits, ids, logps, finish

    def prepare(self, cache, suffix):
        self.restore_rope()
        return self.advance(self.model, self.fork(cache), self.ids(suffix))

    def matcher(self, grammar, prefix=()):
        m = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        for token in prefix:
            if not m.accept_token(int(token)):
                raise RuntimeError("original student prefix rejected by identical grammar")
        return m

    def distribution(self, logits, matcher):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite unmasked logits")
        scores = logits.float().clone()
        mask = xgr.allocate_token_bitmask(1, self.model.config.text_config.vocab_size)
        xgr.reset_token_bitmask(mask)
        if matcher.fill_next_token_bitmask(mask):
            xgr.apply_token_bitmask_inplace(scores, mask.to(scores.device),
                vocab_size=self.model.config.text_config.vocab_size)
        logp = torch.log_softmax(scores, dim=-1)
        if not torch.isfinite(logp).any():
            raise FloatingPointError("empty grammar support")
        return logp

    def decode(self, prepared, seed, *, greedy=False, prefix=(), grammar=None, limit=48):
        self.restore_rope()
        grammar = self.box_grammar if grammar is None else grammar
        cache, logits = self.fork(prepared[0]), prepared[1]
        matcher = self.matcher(grammar)
        ids, logps = [], []
        # Forced original tokens consume the grammar and cache in the same path
        # as sampled tokens, including the comma and its next-token prediction.
        for token in prefix:
            if not matcher.accept_token(int(token)):
                raise RuntimeError("forced prefix is outside bbox grammar")
            ids.append(int(token))
            cache, logits = self.advance(self.model, cache, [token])
        rng = torch.Generator(device="cuda").manual_seed(seed)
        for _ in range(max(0, limit - len(ids))):
            if matcher.is_completed():
                break
            logp = self.distribution(logits, matcher)
            token = int(logp.argmax(-1).item()) if greedy else int(torch.multinomial(logp.exp(), 1, generator=rng).item())
            if not matcher.accept_token(token):
                raise RuntimeError("sample rejected by grammar")
            ids.append(token)
            logps.append(float(logp[0,token]))
            if matcher.is_completed():
                break
            cache, logits = self.advance(self.model, cache, [token])
        return dict(token_ids=ids, text=self.tokenizer.decode(ids, skip_special_tokens=False),
                    generated_logprobs=logps, completed=bool(matcher.is_completed()), seed=seed,
                    forced_prefix_ids=list(prefix))

    def score(self, prepared, sequence, meta, out):
        self.restore_rope()
        cache, logits = self.fork(prepared[0]), prepared[1]
        matcher = self.matcher(self.box_grammar)
        for pos, token in enumerate(sequence):
            logp = self.distribution(logits, matcher)
            support = torch.where(torch.isfinite(logp[0]))[0]
            values = logp[0,support]
            record = dict(meta, token_position=pos, prefix_ids=sequence[:pos],
                          student_token_id=token, student_logprob=float(logp[0,token]),
                          support_ids=support.cpu().tolist(), support_logprobs=values.cpu().tolist(),
                          entropy=float(-(values.exp()*values).sum()))
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            if not matcher.accept_token(token):
                raise RuntimeError("saved student sequence outside grammar")
            if matcher.is_completed():
                if pos != len(sequence)-1:
                    raise RuntimeError("student tokens after grammar completion")
                break
            cache, logits = self.advance(self.model, cache, [token])

    def unchanged(self):
        if [p._version for p in self.model.parameters()] != self.parameter_versions:
            raise RuntimeError("frozen parameter versions changed")


def bbox_record(base, result, *, stage, arm, mode, draw, gt, prefix_coordinates=None):
    box, valid, error = parse_box(result["text"], result["completed"])
    return dict(base, type="bbox", stage=stage, arm=arm, mode=mode, draw=draw,
                prefix_coordinates=prefix_coordinates, **result, bbox=box, valid=valid,
                parse_error=error, iou=iou(box, gt, valid))


def trajectory(engine, row, ti, run_dir):
    inputs = model_input(row)
    if row.get("image_sha256") and file_hash(inputs["image_path"]) != row["image_sha256"]:
        raise RuntimeError("image bytes differ from the frozen dataset manifest")
    base = dict(sample_id=row["sample_id"], image_id=row["image_id"], trajectory_index=ti,
                task_type=row["task_type"])
    stem = hashlib.sha256(row["sample_id"].encode()).hexdigest()[:16] + f"_t{ti}"
    journal = run_dir / "records" / (stem + ".jsonl")
    if journal.exists():
        raise FileExistsError(f"never overwrite an existing trajectory: {journal}")
    start = time.monotonic()
    c0, logits0, prompt_ids, rendered = engine.prefill(inputs)
    c0_hash = cache_fingerprint(c0)
    seed = stable_seed(row["sample_id"], ti, "reasoning")
    cT, logitsT, rids, logps, finish = engine.rollout(c0, logits0, seed,
        progress=lambda n: print(f"REASONING {row['sample_id']} t={ti} tokens={n}", flush=True))
    boundaries = sentence_offsets(engine.tokenizer, rids)
    k = early_offset(len(rids), boundaries)
    # Save the expensive sampled trajectory immediately. The completion record
    # and sidecar carry bridge status without mutating this original record.
    record = dict(base, type="trajectory", seed=seed, prompt_token_ids=prompt_ids,
                  rendered_prompt=rendered, reasoning_ids=rids, reasoning_logprobs=logps,
                  reasoning_text=engine.tokenizer.decode(rids, skip_special_tokens=False),
                  reasoning_length=len(rids), finish_reason=finish, early_offset=k,
                  early_sentence_boundaries=[b for b in boundaries if b <= len(rids)//4],
                  entity="", entity_status="not_generated")
    with (run_dir/"trajectories"/(stem+".json")).open("x") as raw:
        json.dump(record, raw, ensure_ascii=False, allow_nan=False)
        raw.write("\n")
    # Entity generation is independent from natural reasoning; no evaluation
    # field is passed to prepare/decode or the bridge prompt.
    if finish == "stop":
        raw = engine.decode(engine.prepare(cT, ENTITY_PROMPT),
            stable_seed(row["sample_id"], ti, "entity"), greedy=True,
            grammar=engine.entity_grammar, limit=96)
        valid_entity = raw["completed"] and re.fullmatch(ENTITY_REGEX, raw["text"]) is not None
        entity = raw["text"][:-1].strip() if valid_entity else ""
        entity_status = "invalid" if not entity else "unresolved" if entity.casefold()=="unresolved" else "usable"
        record.update(entity=entity, entity_status=entity_status, entity_generation=raw)
    append(journal, record)
    print(f"TRAJECTORY {row['sample_id']} t={ti} T={len(rids)} finish={finish} entity={record['entity_status']}", flush=True)
    if finish != "stop":
        append(journal, dict(base,type="failure",reason="reasoning_"+finish))
        append(journal, dict(base,type="trajectory_done",elapsed_seconds=time.monotonic()-start,
                             inference_complete=False))
        if cache_fingerprint(c0) != c0_hash:
            raise RuntimeError("natural rollout mutated original prompt state")
        return
    # Reconstruct early/late from the exact saved IDs, with sequential GDN
    # advances. This also verifies natural generation versus fixed replay.
    late, replay_logits = engine.replay(c0, rids)
    replay_error = float((replay_logits.float()-logitsT.float()).abs().max()) if rids else 0.
    if replay_error != 0:
        raise RuntimeError(f"natural/replay logits differ: {replay_error}")
    states = {"L0": late}
    if record["entity_status"] == "usable":
        early, _ = engine.replay(c0, rids[:k])
        states.update(L=late,E=early,R=c0)
    state_hashes = {arm:cache_fingerprint(cache) for arm,cache in states.items()}
    prepared = {}
    for arm, cache in states.items():
        suffix = bbox_suffix(inputs["question"], None if arm=="L0" else record["entity"])
        prepared[arm] = engine.prepare(cache, suffix)
    # Compare the entire initial conditional distribution, not only the
    # probabilities of whichever tokens happen to be generated.
    check1 = engine.prepare(late, bbox_suffix(inputs["question"], None))
    if not torch.equal(check1[1], prepared["L0"][1]):
        raise RuntimeError("same-context first-token logits differ")
    del check1
    self_check = engine.decode(prepared["L0"], stable_seed(row["sample_id"],ti,"witness"),greedy=True)
    self_check2 = engine.decode(prepared["L0"], stable_seed(row["sample_id"],ti,"witness"),greedy=True)
    if self_check["token_ids"] != self_check2["token_ids"] or self_check["generated_logprobs"] != self_check2["generated_logprobs"]:
        raise RuntimeError("same-state independent forks are inconsistent")
    late_sequences = []
    for arm in states:
        for mode, draws in (("greedy", 1),("random", 4)):
            for draw in range(draws):
                result = engine.decode(prepared[arm],stable_seed(row["sample_id"],ti,"A",mode,draw),greedy=mode=="greedy")
                rec = bbox_record(base,result,stage="A",arm=arm,mode=mode,draw=draw,gt=row["ground_truth_bbox"])
                append(journal,rec)
                if arm=="L":
                    late_sequences.append(rec)
    if "L" in prepared:
        source = next(r for r in late_sequences if r["mode"]=="random" and r["draw"]==0)
        for nc in (1,2):
            prefix = comma_prefix(engine.tokenizer, source["token_ids"], nc)
            if prefix is None:
                append(journal,dict(base,type="prefix_unavailable",prefix_coordinates=nc,
                                    source_token_ids=source["token_ids"],reason="no_exact_original_token_boundary"))
                continue
            for arm in ("E","L","R"):
                for draw in range(4):
                    result = engine.decode(prepared[arm],stable_seed(row["sample_id"],ti,"B",nc,draw),prefix=prefix)
                    append(journal,bbox_record(base,result,stage="B",arm=arm,mode="random",draw=draw,
                        gt=row["ground_truth_bbox"],prefix_coordinates=nc))
        with gzip.open(run_dir/"distributions"/(stem+".jsonl.gz"), "wt",encoding="utf-8") as out:
            for source in late_sequences:
                for arm in ("E","L","R"):
                    meta=dict(base,arm=arm,source_mode=source["mode"],source_draw=source["draw"])
                    engine.score(prepared[arm],source["token_ids"],meta,out)
    if cache_fingerprint(c0)!=c0_hash or any(cache_fingerprint(states[a])!=h for a,h in state_hashes.items()):
        raise RuntimeError("a measurement branch mutated a saved timeline state")
    engine.unchanged()
    if row.get("image_sha256") and file_hash(inputs["image_path"]) != row["image_sha256"]:
        raise RuntimeError("image bytes changed during inference")
    append(journal,dict(base,type="trajectory_done",inference_complete=True,
        elapsed_seconds=time.monotonic()-start,parity=dict(natural_replay_max_abs_diff=replay_error,
        same_context_forks_exact=True,base_states_unchanged=True,parameters_unchanged=True)))
    print(f"DONE {row['sample_id']} t={ti} seconds={time.monotonic()-start:.1f}",flush=True)


def source_receipts(root):
    paths = list((root/"mmgcot_diagnostic").glob("*.py"))
    paths += [root/p for p in (
        "live_kv_probe_prototype/run_hf_fork.py", "reasoning_checkpoints/extractor.py")]
    return {str(p.relative_to(root)):file_hash(p) for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",required=True)
    p.add_argument("--shard-index",type=int,default=0)
    p.add_argument("--num-shards",type=int,default=1)
    p.add_argument("--model",default=MODEL)
    p.add_argument("--sample-limit",type=int)
    p.add_argument("--trajectory-limit",type=int,choices=(3,),default=3)
    p.add_argument("--resume",action="store_true",help="skip only complete immutable trajectory journals")
    args=p.parse_args()
    if not 0<=args.shard_index<args.num_shards:
        p.error("invalid shard")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    os.environ["PYTHONNOUSERSITE"]="1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    global torch,xgr
    import torch
    import xgrammar as xgr
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    root=Path(__file__).resolve().parents[1]
    rows=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    if args.sample_limit is not None:
        rows=rows[:args.sample_limit]
    selected=[row for i,row in enumerate(rows) if i%args.num_shards==args.shard_index]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for sub in ("records","distributions","workers","trajectories"):
        (args.output_dir/sub).mkdir(exist_ok=True)
    worker=args.output_dir/"workers"/f"shard{args.shard_index}"
    receipt_path=worker.with_suffix(".json")
    protocol=frozen_protocol()
    protocol.update(model=args.model)
    protocol_path=args.output_dir/"protocol.json"
    try:
        with protocol_path.open("x") as f:
            json.dump(protocol,f,ensure_ascii=False,indent=2)
    except FileExistsError:
        if json.loads(protocol_path.read_text())!=protocol:
            raise RuntimeError("existing protocol differs")
    receipts=source_receipts(root)
    weights={p.name:file_hash(p) for p in Path(args.model).iterdir()
             if p.is_file() and (p.suffix in (".json",".safetensors",".jinja") or ".safetensors-" in p.name)}
    manifest_hash=file_hash(args.manifest)
    if receipt_path.exists():
        if not args.resume:
            raise FileExistsError("existing worker; use explicit resume only for identical immutable inputs")
        old=json.loads(receipt_path.read_text())
        if old["manifest_sha256"]!=manifest_hash or old["weights"]!=weights or old["source_hashes"]!=receipts:
            raise RuntimeError("resume with changed inputs/source/weights is prohibited")
    else:
        write_json(receipt_path,dict(command=sys.argv,manifest=str(args.manifest.resolve()),
            manifest_sha256=manifest_hash,source_hashes=receipts,weights=weights,
            git_revision=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
            git_status=subprocess.check_output(["git","status","--short"],cwd=root,text=True),
            versions={k:importlib.metadata.version(k) for k in ("torch","transformers","xgrammar","Pillow")},
            selected_sample_ids=[r["sample_id"] for r in selected],device=args.device,
            shard_index=args.shard_index,num_shards=args.num_shards,trajectory_limit=args.trajectory_limit,
            start_time=time.time()))
    engine=Engine(args.model)
    done=0
    with torch.no_grad():
        for row in selected:
            for ti in range(args.trajectory_limit):
                stem=hashlib.sha256(row["sample_id"].encode()).hexdigest()[:16]+f"_t{ti}"
                path=args.output_dir/"records"/(stem+".jsonl")
                if args.resume and path.exists():
                    records=[json.loads(x) for x in path.read_text().splitlines()]
                    if not records or records[-1].get("type")!="trajectory_done":
                        raise RuntimeError(f"partial journal requires explicit recovery, preserved: {path}")
                    done+=1
                    continue
                try:
                    trajectory(engine,row,ti,args.output_dir)
                    done+=1
                except Exception as exc:
                    append(args.output_dir/"workers"/f"shard{args.shard_index}_errors.jsonl",
                           dict(sample_id=row["sample_id"],trajectory_index=ti,error=repr(exc),time=time.time()))
                    raise
                write_json(worker.with_suffix(".progress.json"),dict(done=done,total=len(selected)*args.trajectory_limit,
                           sample_id=row["sample_id"],trajectory_index=ti,time=time.time()))
    engine.unchanged()
    # Verify only runner/imported inference sources; independent analysis code
    # may legitimately be developed while the pilot is running.
    end_receipts=source_receipts(root)
    runtime_files=("mmgcot_diagnostic/run.py","mmgcot_diagnostic/protocol.py",
                   "live_kv_probe_prototype/run_hf_fork.py","reasoning_checkpoints/extractor.py")
    if any(receipts[k]!=end_receipts[k] for k in runtime_files):
        raise RuntimeError("inference source changed during run")
    write_json(worker.with_suffix(".complete.json"),dict(done=done,time=time.time(),parameters_unchanged=True,
        runtime_sources_unchanged=True,peak_cuda_allocated=torch.cuda.max_memory_allocated()))


if __name__=="__main__":
    main()
