from __future__ import annotations

import json
from pathlib import Path

from training import train_fastconformer_phoneme_modal as modal_train


def test_modal_launch_command_matches_autonomy_contract() -> None:
    assert modal_train.modal_launch_command("quick-smoke") == [
        "modal",
        "run",
        "--detach",
        "training/train_fastconformer_phoneme_modal.py",
        "--job-name",
        "quick-smoke",
    ]


def test_config_from_args_defaults_output_to_modal_volume() -> None:
    args = modal_train.parse_args(["--job-name", "reviewable-smoke", "--max-steps", "7"])
    config = modal_train.config_from_args(args)

    assert config.job_name == "reviewable-smoke"
    assert config.max_steps == 7
    assert config.output_dir == "/outputs/reviewable-smoke"
    assert config.export_onnx is True


def test_local_blockers_allows_modal_volume_paths() -> None:
    config = modal_train.TrainingJobConfig(
        train_manifest="/data/manifests/train.jsonl",
        val_manifest="/data/manifests/val.jsonl",
        phoneme_vocab="/data/phoneme_vocab.txt",
    )

    assert modal_train.local_blockers(config) == []


def test_local_blockers_validates_local_paths(tmp_path: Path) -> None:
    train_manifest = tmp_path / "train.jsonl"
    val_manifest = tmp_path / "val.jsonl"
    vocab = tmp_path / "phonemes.txt"
    train_manifest.write_text('{"audio_filepath":"a.wav","text":"aa"}\n')
    val_manifest.write_text('{"audio_filepath":"b.wav","text":"bb"}\n')
    vocab.write_text("a\nb\n")

    config = modal_train.TrainingJobConfig(
        train_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
        phoneme_vocab=str(vocab),
        batch_size=2,
    )

    assert modal_train.local_blockers(config) == []


def test_generated_training_script_contains_train_export_steps() -> None:
    script = modal_train.build_nemo_training_script()

    assert "EncDecCTCModel" in script
    assert "model.change_vocabulary" in script
    assert "trainer.fit(model)" in script
    assert "model.save_to" in script
    assert "model.export" in script


def test_cli_dry_run_prints_json_plan(capsys) -> None:
    modal_train.cli(["--job-name", "dry-run", "--no-export-onnx"])

    out = capsys.readouterr().out
    plan_text = out.split("\n[training] detached launch:\n", 1)[0]
    plan = json.loads(plan_text)
    assert plan["config"]["job_name"] == "dry-run"
    assert plan["config"]["export_onnx"] is False
    assert "modal run --detach training/train_fastconformer_phoneme_modal.py" in out
