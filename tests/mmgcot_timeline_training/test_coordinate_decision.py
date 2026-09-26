import torch

from mmgcot_timeline_training.coordinate_decision import classify, reverse_kl_on_support


def test_coordinate_decision_is_action_independent():
    for prefix in ("49", "100", "1", "1000"):
        result = classify(prefix, ["0", "3", ","])
        assert result.supervise and result.decision_type == "digit_or_end"
    assert classify("0", [","]).decision_type == "forced_separator"
    assert not classify("0", [","]).supervise
    assert not classify("999", [","]).supervise
    assert classify("49,10,20,45", ["0", "]}"]).coordinate_index == 3
    assert classify("49,10,20,45]", ["}</answer>"]).decision_type == "format"
    assert not classify("49,10,20,45", ["]}</answer>"]).supervise


def test_merged_closing_token_is_end_choice():
    result = classify("1,2,3,4", ["0", "]}</answer>"])
    assert result.supervise and result.decision_type == "digit_or_end"


def test_full_support_reverse_kl_and_added_ending_term_keep_numeric_weight():
    torch.manual_seed(4)
    student = torch.randn(3, 4, dtype=torch.float32, requires_grad=True)
    teacher = torch.randn(3, 4, dtype=torch.float32)
    numeric = [True, False, True]
    decision = [True, True, True]
    values = [reverse_kl_on_support(student[j], teacher[j]) for j in range(3)]
    old_numeric = (values[0] + values[2]) / 2
    new_numeric = sum(v for v, n in zip(values, numeric) if n) / 2
    new_total = sum(v for v, m in zip(values, decision) if m) / 2
    torch.testing.assert_close(old_numeric, new_numeric, atol=0, rtol=0)
    torch.testing.assert_close(new_total - old_numeric, values[1] / 2, atol=1e-6, rtol=1e-6)
    reference = sum(torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=student[j]),
        torch.distributions.Categorical(logits=teacher[j])) for j in range(3)) / 2
    torch.testing.assert_close(new_total, reference, atol=1e-6, rtol=1e-6)
    computed_grad = torch.autograd.grad(new_total, student, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference, student)[0]
    torch.testing.assert_close(computed_grad, reference_grad, atol=1e-6, rtol=1e-6)
