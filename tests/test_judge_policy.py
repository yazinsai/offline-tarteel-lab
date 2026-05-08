from lab_tools.judge_policy import JudgeInput, judge


def test_judge_accepts_improvement_with_tier3():
    out = judge(
        JudgeInput(
            target_recall=0.9,
            target_precision=0.8,
            target_seq_exact=None,
            baseline_recall=0.85,
            baseline_precision=0.75,
            blind_recall=0.88,
            blind_baseline_recall=0.87,
            max_onnx_mb=200,
            onnx_mb=131,
            tier3_completed=True,
        ),
    )
    assert out["accept"] is True
    assert not out["reasons"]


def test_judge_rejects_without_tier3():
    out = judge(
        JudgeInput(
            target_recall=0.99,
            baseline_recall=0.5,
            blind_recall=0.99,
            blind_baseline_recall=0.5,
            target_precision=None,
            baseline_precision=None,
            max_onnx_mb=200,
            onnx_mb=10,
            tier3_completed=False,
        ),
    )
    assert out["accept"] is False
    assert "tier3_required" in out["reasons"]
