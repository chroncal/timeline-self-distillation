import math

import pytest
import torch

from timeline_self_distillation.run_visual_read_diagnostic import add_visual_bias


def test_only_original_image_keys_are_biased_and_mask_is_not_mutated():
    hidden = torch.zeros(1, 1, 8)
    mask = torch.zeros(1, 1, 1, 12)
    out = add_visual_bias(mask, hidden, 12, [2, 3, 4], math.log(4))
    expected = mask.clone()
    expected[..., 2:5] = math.log(4)
    torch.testing.assert_close(out, expected)
    assert torch.count_nonzero(mask) == 0
    torch.testing.assert_close(add_visual_bias(None, hidden, 12, [2], 0), mask)


def test_visual_probe_rejects_prefill_and_out_of_cache_positions():
    with pytest.raises(ValueError, match="single-token"):
        add_visual_bias(None, torch.zeros(1, 2, 8), 12, [2], 1)
    with pytest.raises(ValueError, match="outside"):
        add_visual_bias(None, torch.zeros(1, 1, 8), 12, [12], 1)
