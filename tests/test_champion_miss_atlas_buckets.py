"""Unit tests for champion miss atlas heuristics."""

import pytest

from lab_tools import miss_atlas


@pytest.fixture(scope="module")
def atlas():
    return miss_atlas


@pytest.mark.parametrize(
    ("hit", "gold", "covers"),
    [
        (
            {"surah": 2, "ayah": 140, "ayah_end": 143},
            (2, 143),
            True,
        ),
        (
            {"surah": 2, "ayah": 143, "ayah_end": None},
            (2, 143),
            True,
        ),
        (
            {"surah": 2, "ayah": 140, "ayah_end": 142},
            (2, 143),
            False,
        ),
    ],
)
def test_covers_expected_span(atlas, hit, gold, covers):
    assert atlas.covers_expected(hit, gold) is covers


def test_classification_decode_miss(atlas):
    bucket, _tags = atlas.classify_failure(
        gold=(1, 1),
        pred=(2, 2),
        predicted_ayah_end=None,
        winning_top_ranked=[],
        gold_in_any_hypothesis_pool=False,
        gold_in_winning_hypothesis_pool=False,
        pool_has_span_cover_in_winning_shortlist=False,
    )
    assert bucket == "decode_or_candidate_miss"


def test_classification_rank_within_shortlist(atlas):
    tops = [{"surah": 1, "ayah": 1, "score": 0.9}, {"surah": 1, "ayah": 2, "score": 0.88}]
    bucket, tags = atlas.classify_failure(
        gold=(1, 2),
        pred=(1, 1),
        predicted_ayah_end=None,
        winning_top_ranked=tops,
        gold_in_any_hypothesis_pool=True,
        gold_in_winning_hypothesis_pool=True,
        pool_has_span_cover_in_winning_shortlist=False,
    )
    assert bucket == "ranking_error_within_winning_hypothesis_shortlist"
    assert "narrow_score_gap_top2" in tags
