from mmgcot_diagnostic.protocol import (
    bbox_suffix, comma_prefix, early_offset, iou, model_input, parse_box, stable_seed,
)


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join({1:"12", 2:",", 3:"34", 4:",56", 5:","}[i] for i in ids)


def test_original_token_boundaries_and_unavailable_merged_comma():
    assert comma_prefix(Tokenizer(), [1,2,3,4,5], 1) == [1,2]
    assert comma_prefix(Tokenizer(), [1,2,3,4,5], 2) is None


def test_early_not_best_or_past_quarter():
    assert early_offset(100, [8, 20, 27]) == 20
    assert early_offset(100, [27]) == 25
    assert early_offset(3, []) == 0


def test_inference_annotation_firewall():
    row = dict(question="What color?", image_path="x", reference_cot="SECRET", ground_truth_bbox=[.1]*4)
    assert set(model_input(row)) == {"question", "image_path"}
    assert "SECRET" not in bbox_suffix(row["question"], "a hat")


def test_invalid_box_retained_without_repair():
    b, valid, reason = parse_box("87,458,796,105]}</answer>")
    assert b == [87,458,796,105] and not valid
    assert iou(b, [0,0,1,1], valid) == 0
    assert not parse_box("0,0,1001,1000]}</answer>")[1]
    assert parse_box("0,0,1000,1000]}</answer>")[1]
    assert iou([0,0,1000,1000], [0,0,1,1], True) == 1


def test_seeds_bound_to_identity_not_shard_or_branch():
    assert stable_seed("sample", 0, "A", 1) == stable_seed("sample", 0, "A", 1)
    assert stable_seed("sample", 0) != stable_seed("sample", 1)
