"""Offline champion miss atlas (evaluation-side diagnostics).

Reads benchmark JSON + audio; does not expose labels to phoneme matcher inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab_tools.paths import lab_root


def _load_joint_module():
    jp = lab_root() / "experiments" / "phoneme_matcher_joint02" / "run.py"
    spec = importlib.util.spec_from_file_location("phoneme_joint02_inner", jp)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load phoneme joint from {jp}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_joint = None


def _joint_mod():
    global _joint
    if _joint is None:
        _joint = _load_joint_module()
    return _joint


def _expected_first_sa(sample: dict[str, Any]) -> tuple[int, int] | None:
    ev = sample.get("expected_verses")
    if isinstance(ev, list) and ev:
        first = ev[0]
        if isinstance(first, dict):
            try:
                return int(first.get("surah", 0)), int(first.get("ayah", 0))
            except (TypeError, ValueError):
                return None
    return None


def _norm_pair(hit: dict) -> tuple[int, int]:
    return int(hit["surah"]), int(hit["ayah"])


def covers_expected(hit: dict, gold: tuple[int, int]) -> bool:
    gs, ga = gold
    if int(hit["surah"]) != gs:
        return False
    a0 = int(hit["ayah"])
    end = hit.get("ayah_end")
    a1 = int(end) if end is not None else a0
    if a1 < a0:
        a0, a1 = a1, a0
    return a0 <= ga <= a1


def classify_failure(
    *,
    gold: tuple[int, int],
    pred: tuple[int, int],
    predicted_ayah_end: int | None,
    winning_top_ranked: list[dict],
    gold_in_any_hypothesis_pool: bool,
    gold_in_winning_hypothesis_pool: bool,
    pool_has_span_cover_in_winning_shortlist: bool,
) -> tuple[str, list[str]]:
    tags: list[str] = []

    if predicted_ayah_end is not None:
        try:
            pae = int(predicted_ayah_end)
        except (TypeError, ValueError):
            pae = None
        if pae is not None and pae != pred[1]:
            tags.append("predicted_span_ayah_range")

    if pool_has_span_cover_in_winning_shortlist and pred != gold and predicted_ayah_end is not None:
        tags.append("possible_multi_verse_span_or_boundary")

    if not gold_in_any_hypothesis_pool:
        return "decode_or_candidate_miss", tags

    if int(pred[0]) != int(gold[0]):
        tags.append("predicted_cross_surah_mismatch")

    if gold_in_winning_hypothesis_pool and pred != gold:
        if len(winning_top_ranked) >= 2:
            s0 = float(winning_top_ranked[0]["score"])
            s1 = float(winning_top_ranked[1]["score"])
            if abs(s0 - s1) <= 0.02 + 1e-6:
                tags.append("narrow_score_gap_top2")
        return "ranking_error_within_winning_hypothesis_shortlist", tags

    if gold_in_any_hypothesis_pool and not gold_in_winning_hypothesis_pool:
        tags.append("gold_only_in_secondary_hypothesis")
        return "ranking_error_across_hypotheses", tags

    return "ranking_residual", tags


def diagnose_clip(joint_mod, audio_abs: Path, gold: tuple[int, int]) -> dict[str, Any]:
    joint_mod._ensure_loaded()
    assert joint_mod._onnx_session is not None
    audio = joint_mod._load_audio(str(audio_abs))
    logprobs = joint_mod._compute_logprobs(audio, joint_mod._onnx_session)
    hyps = joint_mod._hypotheses_from_logprobs(logprobs)
    top_k = getattr(joint_mod, "TOP_K_LEVENSHTEIN", 18)

    per_hyp_candidates: dict[str, list[dict]] = {}
    pool_best_cover_score = 0.0

    winner_hyp = ""
    best_rank = -1.0
    winning_top_ranked: list[dict] = []
    union_has_cover = False
    predicted_hit: dict | None = None

    for hyp in hyps:
        tops = joint_mod._match_phoneme_text(hyp, top_k=top_k)
        tops_u = [{**t, "_pair": _norm_pair(t)} for t in tops]
        per_hyp_candidates[hyp[:180]] = tops_u[:top_k]

        if any(covers_expected(t, gold) for t in tops):
            union_has_cover = True

        cover_score = max(
            (float(t["score"]) for t in tops if covers_expected(t, gold)),
            default=0.0,
        )
        pool_best_cover_score = max(pool_best_cover_score, cover_score)

        if tops:
            ts = float(tops[0]["score"])
            if ts > best_rank:
                best_rank = ts
                winner_hyp = hyp
                winning_top_ranked = tops[: max(top_k, 10)]
                predicted_hit = tops[0]

    greedy = joint_mod._labels_to_phoneme_string(joint_mod._greedy_decode_ids(logprobs))

    if not predicted_hit:
        pred_pair = (0, 0)
        predicted_ayah_end = None
        if not winner_hyp:
            winner_hyp = greedy
    else:
        pred_pair = int(predicted_hit["surah"]), int(predicted_hit["ayah"])
        predicted_ayah_end = predicted_hit.get("ayah_end")
        if predicted_ayah_end is not None:
            try:
                predicted_ayah_end = int(predicted_ayah_end)
            except (TypeError, ValueError):
                predicted_ayah_end = None

    gold_in_any_hypothesis_pool = union_has_cover
    gold_in_winning_hypothesis_pool = any(covers_expected(t, gold) for t in winning_top_ranked)

    winning_span_cover_flag = False
    for t in winning_top_ranked:
        end = t.get("ayah_end")
        if end is None:
            continue
        try:
            if int(end) != int(t["ayah"]) and covers_expected(t, gold):
                winning_span_cover_flag = True
                break
        except (TypeError, ValueError):
            continue

    gold_match = gold == pred_pair
    bucket, tags = (
        ("correct", [])
        if gold_match
        else classify_failure(
            gold=gold,
            pred=pred_pair,
            predicted_ayah_end=predicted_ayah_end,
            winning_top_ranked=winning_top_ranked,
            gold_in_any_hypothesis_pool=gold_in_any_hypothesis_pool,
            gold_in_winning_hypothesis_pool=gold_in_winning_hypothesis_pool,
            pool_has_span_cover_in_winning_shortlist=winning_span_cover_flag,
        )
    )

    top_gaps = []
    if len(winning_top_ranked) >= 2:
        top_gaps.append(
            float(winning_top_ranked[0]["score"]) - float(winning_top_ranked[1]["score"]),
        )

    primary_pool_detail = [
        {"surah": t["surah"], "ayah": t["ayah"], "score": t["score"], "ayah_end": t.get("ayah_end")}
        for t in winning_top_ranked[:8]
    ]

    return {
        "gold_surah": gold[0],
        "gold_ayah": gold[1],
        "predicted_surah": pred_pair[0],
        "predicted_ayah": pred_pair[1],
        "predicted_ayah_end": predicted_ayah_end,
        "winning_hypothesis_trim": winner_hyp[:200],
        "greedy_trim": greedy[:200],
        "beam_hypotheses_n": len(hyps),
        "pool_best_cover_score_for_gold": pool_best_cover_score,
        "gold_in_union_pool_hypotheses": gold_in_any_hypothesis_pool,
        "gold_covers_candidate_in_winning_shortlist": gold_in_winning_hypothesis_pool,
        "matcher_hypothesis_count": len(per_hyp_candidates),
        "matcher_candidate_rows_summed_over_hyps": sum(len(v) for v in per_hyp_candidates.values()),
        "top1_top2_gap": top_gaps[0] if top_gaps else None,
        "primary_top_candidates": primary_pool_detail,
        "classification_bucket": bucket,
        "tags": tags,
    }


def build_atlas(*, corpus: str, limit: int = 0) -> dict[str, Any]:
    root = lab_root()
    bench = root / "benchmark" / corpus
    man_name = "manifest" + ".json"
    manifest_path = bench / man_name
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples: list = list(manifest.get("samples") or [])
    manifest_n = len(samples)
    if limit > 0:
        samples = samples[:limit]

    joint_mod = _joint_mod()
    misses: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = defaultdict(int)
    correct_n = 0

    for s in samples:
        sid = str(s.get("id", "?"))
        rel = str(s.get("file", ""))
        ap = bench / rel
        gold = _expected_first_sa(s)
        row_base = {
            "id": sid,
            "benchmark_file": rel,
            "manifest_category": s.get("category"),
            "manifest_reciter": s.get("reciter"),
            "manifest_source": s.get("source"),
        }

        if gold is None:
            misses.append({**row_base, "error": "missing_expected_verses"})
            bucket_counts["manifest_error"] += 1
            continue
        try:
            quick = joint_mod.predict(str(ap))
            pq = (int(quick.get("surah", 0)), int(quick.get("ayah", 0)))
            if pq == gold:
                correct_n += 1
                continue
            detail = diagnose_clip(joint_mod, ap, gold)
        except Exception as exc:  # noqa: BLE001
            misses.append({**row_base, "exception": repr(exc)})
            bucket_counts["runtime_error"] += 1
            continue

        if detail["classification_bucket"] == "correct":
            correct_n += 1
            continue

        misses.append({**row_base, **detail})
        bucket_counts[detail["classification_bucket"]] += 1

    return {
        "schema": "offline-tarteel.champion_miss_atlas.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "champion_experiment": "phoneme_matcher_joint02",
        "corpus": corpus,
        "manifest_samples": manifest_n,
        "evaluated_rows": len(samples),
        "correct_count": correct_n,
        "miss_count": len(misses),
        "bucket_histogram": dict(bucket_counts),
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build champion miss atlas JSON")
    ap.add_argument("--corpus", default="test_corpus_v3")
    ap.add_argument("--limit", type=int, default=0, help="0 = full manifest order")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Defaults to artifacts/runs/champion_miss_atlas_<timestamp>.json",
    )
    args = ap.parse_args()

    atlas = build_atlas(corpus=args.corpus, limit=args.limit)
    out = args.out
    if out is None:
        stem = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = lab_root() / "artifacts" / "runs" / f"champion_miss_atlas_{stem}.json"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(atlas, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote_atlas": str(out), "miss_count": atlas["miss_count"]}))
    histogram = atlas.get("bucket_histogram")
    print("bucket_histogram", histogram, file=sys.stderr)


if __name__ == "__main__":
    main()
