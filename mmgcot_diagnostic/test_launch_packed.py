import pytest

from mmgcot_diagnostic.launch_packed import assignments


def test_assignments_are_disjoint_stable_shards():
    assert assignments(["1", "2", "3"], 2) == [
        (0, "1", 0), (1, "2", 0), (2, "3", 0),
        (3, "1", 1), (4, "2", 1), (5, "3", 1),
    ]


@pytest.mark.parametrize("gpus,workers", [([], 1), (["1", "1"], 1), (["1"], 0)])
def test_invalid_layout_rejected(gpus, workers):
    with pytest.raises(ValueError):
        assignments(gpus, workers)
