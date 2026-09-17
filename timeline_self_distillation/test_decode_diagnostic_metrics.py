import pytest

from timeline_self_distillation.decode_diagnostic_metrics import first_numeric_difference, first_token_divergence


def test_first_changed_digit_is_reported_by_coordinate_not_shifted_flat_string():
    diff = first_numeric_difference("47,20,900,800]}</answer>", "47,28,901,800]}</answer>")
    assert diff["different"]
    assert diff["coordinate"] == "y_min"
    assert diff["digit_index"] == 2
    assert (diff["base_char"], diff["checkpoint_char"]) == ("0", "8")


@pytest.mark.parametrize("a,b,chars", [("47", "470", ("<END>", "0")), ("470", "47", ("0", "<END>"))])
def test_ending_a_number_is_a_numeric_difference(a, b, chars):
    diff = first_numeric_difference(f"{a},20,900,800]}}</answer>", f"{b},20,900,800]}}</answer>")
    assert diff["coordinate"] == "x_min"
    assert diff["digit_index"] == 3
    assert diff["kind"] == "value_length"
    assert (diff["base_char"], diff["checkpoint_char"]) == chars


def test_equal_coordinates_have_no_numeric_difference_even_if_suffix_differs():
    diff = first_numeric_difference("1,2,3,4]}</answer>", "1,2,3,4]")
    assert diff["comparable"] and not diff["different"]


def test_invalid_geometry_is_still_digit_comparable():
    diff = first_numeric_difference("87,458,796,943]}</answer>", "87,458,796,105]}</answer>")
    assert diff["coordinate"] == "y_max"
    assert diff["digit_index"] == 1


def test_incomplete_numeric_output_is_reported_not_assumed_unchanged():
    diff = first_numeric_difference("1,2,3,4]", "1,2,")
    assert not diff["comparable"]
    assert diff["different"] is None


class Tokenizer:
    pieces = {1: "4", 2: "7", 3: ",", 4: "0", 5: "2", 6: "]}</", 7: "]}"}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


def test_token_divergence_catches_comma_vs_continuation_at_same_prefix():
    diff = first_token_divergence([1, 2, 3, 5], [1, 2, 4, 3, 5], Tokenizer())
    assert diff["different"]
    assert diff["token_position"] == 3
    assert diff["shared_prefix"] == "47"
    assert diff["kind"] == "numeric_termination"


def test_format_only_token_divergence_is_not_called_a_digit_change():
    diff = first_token_divergence([1, 3, 5, 6], [1, 3, 5, 7], Tokenizer())
    assert diff["kind"] == "format_or_ending"


def test_identical_ids_are_explicitly_reported():
    assert first_token_divergence([1, 2], [1, 2], Tokenizer())["different"] is False
