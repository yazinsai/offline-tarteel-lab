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
    assert config.precision == "16"
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
    assert "import pytorch_lightning as pl" in script
    assert "model.change_vocabulary" in script
    assert '"name": "adamw"' in script
    assert "trainer.fit(model)" in script
    assert "model.save_to" in script
    assert "model.export" in script


def test_modal_image_pins_binary_data_stack_before_nemo_install() -> None:
    source = Path(modal_train.__file__).read_text()
    assert '"Cython", "datasets>=2.18", "numpy<2", "pyarrow>=14", "wheel"' in source
    assert '"nemo_toolkit[asr]==1.23.0"' in source
    assert '"pytorch-lightning==2.0.7"' in source


def test_hf_secret_defaults_to_optional() -> None:
    assert modal_train.HF_SECRET_NAME == ""


def test_cli_dry_run_prints_json_plan(capsys) -> None:
    modal_train.cli(["--job-name", "dry-run", "--no-export-onnx"])

    out = capsys.readouterr().out
    plan_text = out.split("\n[training] detached launch:\n", 1)[0]
    plan = json.loads(plan_text)
    assert plan["config"]["job_name"] == "dry-run"
    assert plan["config"]["export_onnx"] is False
    assert "modal run --detach training/train_fastconformer_phoneme_modal.py" in out


def test_launch_spawns_remote_training_without_blocking(monkeypatch, capsys) -> None:
    calls = []

    class FakeCall:
        object_id = "fc-123"

    class FakeTrainRemote:
        def remote(self, _payload):
            raise AssertionError("detached training must not use remote()")

        def spawn(self, payload):
            calls.append(payload)
            return FakeCall()

    monkeypatch.setattr(modal_train, "modal", object())
    monkeypatch.setattr(modal_train, "train_remote", FakeTrainRemote())

    modal_train.launch(modal_train.TrainingJobConfig(job_name="spawned-smoke"))

    out = json.loads(capsys.readouterr().out)
    assert calls[0]["job_name"] == "spawned-smoke"
    assert out == {
        "job_name": "spawned-smoke",
        "output_dir": "/outputs/fastconformer-phoneme-smoke",
        "function_call_id": "fc-123",
        "status": "spawned",
    }
