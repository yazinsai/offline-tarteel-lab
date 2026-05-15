"""Modal entrypoint for phoneme FastConformer training.

This file intentionally owns only the cloud training plumbing. It does not
commit model artifacts and it does not touch runtime experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import textwrap
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

APP_NAME = "offline-tarteel-fastconformer-phoneme"
DATA_VOLUME_NAME = os.getenv("TARTEEL_MODAL_DATA_VOLUME", "offline-tarteel-lab-data")
OUTPUT_VOLUME_NAME = os.getenv("TARTEEL_MODAL_OUTPUT_VOLUME", "offline-tarteel-lab-models")
HF_SECRET_NAME = os.getenv("TARTEEL_MODAL_HF_SECRET", "huggingface-secret")

DATA_ROOT = Path("/data")
OUTPUT_ROOT = Path("/outputs")
DEFAULT_TRAIN_MANIFEST = str(DATA_ROOT / "manifests" / "train.jsonl")
DEFAULT_VAL_MANIFEST = str(DATA_ROOT / "manifests" / "val.jsonl")
DEFAULT_PHONEME_VOCAB = str(DATA_ROOT / "phoneme_vocab.txt")
DEFAULT_BASE_MODEL = "nvidia/stt_en_fastconformer_ctc_large"
DEFAULT_GPU = "A10G"
DEFAULT_TIMEOUT_SECONDS = 12 * 60 * 60


@dataclass(frozen=True)
class TrainingJobConfig:
    job_name: str = "fastconformer-phoneme-smoke"
    train_manifest: str = DEFAULT_TRAIN_MANIFEST
    val_manifest: str = DEFAULT_VAL_MANIFEST
    phoneme_vocab: str = DEFAULT_PHONEME_VOCAB
    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = str(OUTPUT_ROOT / "fastconformer-phoneme-smoke")
    max_steps: int = 100
    max_epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    precision: str = "16-mixed"
    export_onnx: bool = True

    def normalized(self) -> "TrainingJobConfig":
        safe_name = self.job_name.strip() or "fastconformer-phoneme-smoke"
        return TrainingJobConfig(
            **{
                **asdict(self),
                "job_name": safe_name,
                "output_dir": self.output_dir.rstrip("/") or str(OUTPUT_ROOT / safe_name),
            }
        )


def modal_launch_command(job_name: str) -> list[str]:
    return [
        "modal",
        "run",
        "--detach",
        "training/train_fastconformer_phoneme_modal.py",
        "--job-name",
        job_name,
    ]


def local_blockers(config: TrainingJobConfig) -> list[str]:
    blockers: list[str] = []
    for label, raw_path in (
        ("train manifest", config.train_manifest),
        ("validation manifest", config.val_manifest),
        ("phoneme vocab", config.phoneme_vocab),
    ):
        path = Path(raw_path)
        if path.is_absolute() and str(path).startswith((str(DATA_ROOT), str(OUTPUT_ROOT))):
            continue
        if not path.exists():
            blockers.append(f"missing {label}: {raw_path}")
    if config.max_steps <= 0 and config.max_epochs <= 0:
        blockers.append("max_steps or max_epochs must be positive")
    if config.batch_size <= 0:
        blockers.append("batch_size must be positive")
    return blockers


def build_nemo_training_script() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        import json
        import os
        from pathlib import Path

        import lightning.pytorch as pl
        from nemo.collections.asr.models import EncDecCTCModel


        def read_vocab(path: str) -> list[str]:
            vocab = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
            if not vocab:
                raise ValueError(f"phoneme vocab is empty: {path}")
            return vocab


        def main() -> None:
            config = json.loads(Path("/tmp/tarteel_training_config.json").read_text())
            output_dir = Path(config["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)

            for required in ("train_manifest", "val_manifest", "phoneme_vocab"):
                if not Path(config[required]).exists():
                    raise FileNotFoundError(f"missing {required}: {config[required]}")

            token = os.getenv("HF_TOKEN")
            if token:
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

            base_model = config["base_model"]
            if base_model.endswith(".nemo") or Path(base_model).exists():
                model = EncDecCTCModel.restore_from(restore_path=base_model)
            else:
                model = EncDecCTCModel.from_pretrained(model_name=base_model)

            model.change_vocabulary(new_vocabulary=read_vocab(config["phoneme_vocab"]))
            model.setup_training_data(
                train_data_config={
                    "manifest_filepath": config["train_manifest"],
                    "sample_rate": 16000,
                    "batch_size": config["batch_size"],
                    "shuffle": True,
                    "num_workers": min(8, os.cpu_count() or 1),
                    "pin_memory": True,
                }
            )
            model.setup_validation_data(
                val_data_config={
                    "manifest_filepath": config["val_manifest"],
                    "sample_rate": 16000,
                    "batch_size": config["batch_size"],
                    "shuffle": False,
                    "num_workers": min(8, os.cpu_count() or 1),
                    "pin_memory": True,
                }
            )
            model.setup_optimization(
                optim_config={
                    "optimizer": "adamw",
                    "lr": config["learning_rate"],
                    "betas": [0.9, 0.98],
                    "weight_decay": 0.001,
                    "sched": {"name": "CosineAnnealing", "warmup_steps": 100},
                }
            )

            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_steps=config["max_steps"],
                max_epochs=config["max_epochs"],
                precision=config["precision"],
                default_root_dir=str(output_dir),
                log_every_n_steps=10,
                enable_checkpointing=True,
            )
            trainer.fit(model)

            nemo_path = output_dir / f"{config['job_name']}.nemo"
            model.save_to(str(nemo_path))
            artifacts = {"nemo": str(nemo_path)}
            if config["export_onnx"]:
                onnx_path = output_dir / f"{config['job_name']}.onnx"
                model.export(str(onnx_path))
                artifacts["onnx"] = str(onnx_path)

            (output_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\\n")


        if __name__ == "__main__":
            main()
        """
    ).strip()


try:
    import modal  # type: ignore[import-not-found]
except ModuleNotFoundError:
    modal = None  # type: ignore[assignment]


if modal is not None:
    app = modal.App(APP_NAME)
    data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
    output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "libsndfile1")
        .pip_install(
            "nemo_toolkit[asr]",
            "torch",
            "torchaudio",
            "lightning",
            "huggingface_hub",
        )
    )

    @app.function(
        image=image,
        gpu=DEFAULT_GPU,
        volumes={DATA_ROOT: data_volume, OUTPUT_ROOT: output_volume},
        secrets=[modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])],
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    def train_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
        data_volume.reload()
        config = TrainingJobConfig(**config_payload).normalized()
        blockers = local_blockers(config)
        if blockers:
            raise RuntimeError("Modal training blockers: " + "; ".join(blockers))

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = Path("/tmp/tarteel_training_config.json")
        script_path = Path("/tmp/tarteel_train_fastconformer_phoneme.py")
        config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")
        script_path.write_text(build_nemo_training_script() + "\n")

        print(f"[training] job={config.job_name}")
        print(f"[training] train_manifest={config.train_manifest}")
        print(f"[training] val_manifest={config.val_manifest}")
        print(f"[training] output_dir={config.output_dir}")
        subprocess.run(["python", str(script_path)], check=True)

        output_volume.commit()
        return {
            "job_name": config.job_name,
            "output_dir": config.output_dir,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

else:
    app = None

    def train_remote(config_payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("modal is not importable in this Python environment")


def print_plan(config: TrainingJobConfig) -> None:
    payload = asdict(config.normalized())
    blockers = local_blockers(config)
    print(json.dumps({"config": payload, "local_blockers": blockers}, indent=2))
    print("[training] detached launch:")
    print(" ".join(shlex.quote(part) for part in modal_launch_command(config.job_name)))


def launch(config: TrainingJobConfig) -> None:
    if modal is None:
        print_plan(config)
        raise SystemExit(
            "Modal SDK is not importable here; install/authenticate Modal or use modal run from a "
            "Modal-capable environment."
        )
    result = train_remote.remote(asdict(config.normalized()))  # type: ignore[attr-defined]
    print(json.dumps(result, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch phoneme FastConformer training on Modal")
    parser.add_argument("--job-name", default=TrainingJobConfig.job_name)
    parser.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--phoneme-vocab", default=DEFAULT_PHONEME_VOCAB)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-steps", type=int, default=TrainingJobConfig.max_steps)
    parser.add_argument("--max-epochs", type=int, default=TrainingJobConfig.max_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingJobConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingJobConfig.learning_rate)
    parser.add_argument("--precision", default=TrainingJobConfig.precision)
    parser.add_argument("--no-export-onnx", action="store_true")
    parser.add_argument("--launch", action="store_true", help="Launch from python -m instead of dry-run")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainingJobConfig:
    output_dir = args.output_dir or str(OUTPUT_ROOT / args.job_name)
    return TrainingJobConfig(
        job_name=args.job_name,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        phoneme_vocab=args.phoneme_vocab,
        base_model=args.base_model,
        output_dir=output_dir,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        precision=args.precision,
        export_onnx=not args.no_export_onnx,
    ).normalized()


def cli(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    if args.launch:
        launch(config)
    else:
        print_plan(config)


def modal_main(
    job_name: str = TrainingJobConfig.job_name,
    train_manifest: str = DEFAULT_TRAIN_MANIFEST,
    val_manifest: str = DEFAULT_VAL_MANIFEST,
    phoneme_vocab: str = DEFAULT_PHONEME_VOCAB,
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = "",
    max_steps: int = TrainingJobConfig.max_steps,
    max_epochs: int = TrainingJobConfig.max_epochs,
    batch_size: int = TrainingJobConfig.batch_size,
    learning_rate: float = TrainingJobConfig.learning_rate,
    precision: str = TrainingJobConfig.precision,
    export_onnx: bool = True,
) -> None:
    config = TrainingJobConfig(
        job_name=job_name,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        phoneme_vocab=phoneme_vocab,
        base_model=base_model,
        output_dir=output_dir or str(OUTPUT_ROOT / job_name),
        max_steps=max_steps,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        precision=precision,
        export_onnx=export_onnx,
    ).normalized()
    launch(config)


if modal is not None:
    modal_main = app.local_entrypoint()(modal_main)


if __name__ == "__main__":
    cli()
