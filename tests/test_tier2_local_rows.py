import json
import sys

from lab_tools import tier2_local


def test_tier2_report_includes_row_level_feedback(tmp_path, monkeypatch):
    corpus = tmp_path / "benchmark" / "test_corpus_v3"
    corpus.mkdir(parents=True)
    (corpus / "a.wav").write_bytes(b"RIFF")
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "sample-a",
                        "file": "a.wav",
                        "category": "short",
                        "expected_verses": [{"surah": 1, "ayah": 1}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    experiment = tmp_path / "experiments" / "demo"
    experiment.mkdir(parents=True)
    (experiment / "run.py").write_text(
        "def predict(_audio_path):\n"
        "    return {'surah': 1, 'ayah': 1, 'score': 0.9, 'transcript': 'demo'}\n",
        encoding="utf-8",
    )
    out = tmp_path / "tier2.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tier2_local",
            "--root",
            str(tmp_path),
            "--corpus",
            "test_corpus_v3",
            "--experiment",
            "demo",
            "--json",
            str(out),
        ],
    )

    tier2_local.main()

    report = json.loads(out.read_text(encoding="utf-8"))
    row = report["results"][0]["rows"][0]
    assert row["id"] == "sample-a"
    assert row["category"] == "short"
    assert row["expected"] == {"surah": 1, "ayah": 1}
    assert row["predicted"]["transcript"] == "demo"
    assert row["correct"] is True
