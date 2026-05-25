from __future__ import annotations

import json
from pathlib import Path


def _sample_by_id() -> dict[str, dict]:
    manifest_path = Path("benchmark/test_corpus_v3/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {sample["id"]: sample for sample in manifest["samples"]}


def test_confirmed_tlog_samples_start_at_filename_verse() -> None:
    samples = _sample_by_id()

    assert samples["tlog_m008_107_001"]["expected_verses"][0] == {"surah": 107, "ayah": 1}
    assert samples["tlog_m059_010_053"]["expected_verses"][0] == {"surah": 10, "ayah": 53}


def test_confirmed_user_ikhlas_sample_starts_at_decoded_verse() -> None:
    sample = _sample_by_id()["user_ikhlas_2_3"]

    assert sample["expected_verses"][0] == {"surah": 112, "ayah": 3}
