"""Select existing reasoning evidence instead of freely rewriting the request.

The model chooses one source sentence; the teacher receives that exact slice.
Source fidelity is mechanically checked, but target correctness is not assumed.
"""

from __future__ import annotations

import re
from typing import Any

from timeline_self_distillation.entity_bridge import ENTITY_REGEX, inspect_bridge_result

_META = re.compile(r"\b(?:user|prompt|coordinate\w*|bounding\s+box|format|response)\b", re.I)


def evidence_candidates(reasoning_text: str) -> list[dict[str, Any]]:
    """Extract complete coordinate-free source clauses without using image/GT.

    Split on lines and sentence-ending whitespace, strip only leading list or
    Markdown heading markers, and retain exact original character offsets.
    Task/answer-format statements and coordinate-bearing clauses are excluded.
    Candidates are neither filtered by target words nor ranked by correctness.
    """
    boundaries = list(re.finditer(r"\n+|(?<=[.!?])\s+", reasoning_text))
    chunks = []
    start = 0
    for boundary in boundaries:
        chunks.append((start, boundary.start()))
        start = boundary.end()
    chunks.append((start, len(reasoning_text)))
    candidates, seen = [], set()
    for start, end in chunks:
        raw = reasoning_text[start:end]
        text = raw.strip()
        text = re.sub(r"^(?:\d+\.\s+|[*-]\s+)", "", text)
        text = re.sub(r"^\*\*[^*]+\*\*\s*", "", text).strip()
        if not text or text.endswith(":") or _META.search(text):
            continue
        if re.fullmatch(ENTITY_REGEX, text + '"') is None:
            continue
        key = text.casefold()
        if key in seen:
            continue
        offset = raw.find(text)
        if offset < 0:
            raise AssertionError("candidate cleaning changed source text")
        source_start = start + offset
        assert reasoning_text[source_start : source_start + len(text)] == text
        candidates.append(
            {"id": len(candidates) + 1, "text": text, "start": source_start, "end": source_start + len(text)}
        )
        seen.add(key)
    return candidates


def selection_prompt(candidates: list[dict[str, Any]]) -> str:
    listing = "\n".join(f"{entry['id']}: {entry['text']}" for entry in candidates)
    return (
        "\nThe completed reasoning selected a particular target instance. "
        "Choose the ONE source sentence below that best identifies that FINAL instance "
        "for a reader who has not seen the reasoning. Prefer the sentence that preserves "
        "its distinguishing appearance, position, or relationship. Do not choose a rejected "
        "alternative or a sentence about a different object. These are verbatim excerpts, "
        "not new evidence. Return only its integer ID. Return 0 if none identifies the final target.\n"
        + listing
        + "\n</think>\n<evidence_id>"
    )


def selected_bridge(raw, candidates, expression, reasoning_text):
    text = str(raw.get("text", ""))
    match = re.fullmatch(r"(\d+)</evidence_id>", text)
    selection = int(match.group(1)) if raw.get("completed", False) and match else None
    candidate = next((item for item in candidates if item["id"] == selection), None)
    entity = "" if candidate is None else candidate["text"]
    source_exact = candidate is not None and reasoning_text[candidate["start"] : candidate["end"]] == entity
    if candidate is not None and not source_exact:
        raise AssertionError("selected entity is not its declared source slice")
    # Reuse only the format/echo checker; it does not infer semantic correctness.
    audit = inspect_bridge_result({"text": entity + '"', "completed": bool(entity)}, expression, "instance_v2")
    audit.update(
        version="extractive_v3",
        semantic_status="source_grounded_needs_target_review" if source_exact else "unresolved",
        source_text_exact=source_exact,
        selected_id=selection,
        selected_evidence=candidate,
        candidate_count=len(candidates),
        selection_completed=bool(raw.get("completed", False)),
        instance_binding_verified=False,
    )
    if not entity:
        audit["issues"].append("no_valid_evidence_selection")
    return {
        "entity": entity,
        "entity_result": {**raw, "output_kind": "source_evidence_id", "candidates": candidates},
        "entity_audit": audit,
    }


def generate_extractive_entity_bridge(model, tokenizer, cache, expression, reasoning_text, seed):
    # Local imports keep the source extraction helper CPU-only and avoid a cycle.
    import xgrammar as xgr

    from timeline_self_distillation.run_teacher_pilot import constrained_generate, fork

    candidates = evidence_candidates(reasoning_text)
    if not candidates:
        return selected_bridge(
            {"text": "0</evidence_id>", "completed": True, "token_ids": []}, candidates, expression, reasoning_text
        )
    choices = "(?:" + "|".join(str(index) for index in range(len(candidates) + 1)) + ")</evidence_id>"
    compiler = xgr.GrammarCompiler(
        xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.text_config.vocab_size)
    )
    raw = constrained_generate(
        model,
        tokenizer,
        compiler.compile_regex(choices),
        fork(cache),
        selection_prompt(candidates),
        seed,
        greedy=True,
        limit=96,
    )
    raw["selection_regex"] = choices
    return selected_bridge(raw, candidates, expression, reasoning_text)
