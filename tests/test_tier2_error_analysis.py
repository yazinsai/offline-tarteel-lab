from __future__ import annotations

from types import SimpleNamespace

from lab_tools.tier2_error_analysis import analyze_miss, incorrect_rows


def test_incorrect_rows_returns_only_rows_with_transcripts() -> None:
    report = {
        "rows": [
            {"index": 1, "correct": True, "predicted": {"transcript": "aaa"}},
            {"index": 2, "correct": False, "predicted": {"transcript": "bbb"}},
            {"index": 3, "correct": False, "predicted": {}},
        ]
    }

    rows = incorrect_rows(report)

    assert [row["index"] for row in rows] == [2]


def test_incorrect_rows_accepts_nested_tier2_result_rows() -> None:
    report = {
        "results": [
            {
                "experiment": "demo",
                "rows": [
                    {"index": 1, "correct": False, "predicted": {"transcript": "aaa"}},
                    {"index": 2, "correct": False, "predicted": None},
                ],
            }
        ]
    }

    rows = incorrect_rows(report)

    assert [row["index"] for row in rows] == [1]


def test_analyze_miss_reports_candidate_groups_and_expected_ranks() -> None:
    row = {
        "index": 7,
        "id": "sample-7",
        "expected": {"surah": 44, "ayah": 1},
        "predicted": {"surah": 69, "ayah": 33, "score": 0.59, "transcript": "decoded phones"},
    }
    sample = {"id": "sample-7", "file": "sample.wav", "category": "multi"}

    def base_candidates(_text: str, top_k: int = 18):
        return [
            {"surah": 69, "ayah": 33, "score": 0.59},
            {"surah": 44, "ayah": 1, "score": 0.57},
        ]

    module = SimpleNamespace(
        _base=SimpleNamespace(_match_phoneme_text=base_candidates),
        _surah_prefix_candidates=lambda _text: [{"surah": 44, "ayah": 1, "score": 0.58}],
        _global_span_candidates=lambda _text: [{"surah": 51, "ayah": 1, "score": 0.55}],
        _same_surah_rewind_candidate=lambda _text, _best: None,
    )

    analysis = analyze_miss(row, sample, module)

    assert analysis["id"] == "sample-7"
    assert analysis["category"] == "multi"
    assert analysis["base"]["expected_rank"] == 2
    assert analysis["prefix"]["expected_rank"] == 1
    assert analysis["global_span"]["expected_rank"] is None
