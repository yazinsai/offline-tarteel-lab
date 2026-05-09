"""Modal training entrypoint for phoneme FastConformer smoke jobs.

Local: ``python -m training.train_fastconformer_phoneme_modal --job-name NAME``
Modal: ``modal run --detach training/train_fastconformer_phoneme_modal.py --job-name NAME``
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any

import modal

app = modal.App("offline-tarteel-fastconformer-phoneme")

image = modal.Image.debian_slim(python_version="3.12").pip_install()


def train(job_name: str) -> dict[str, Any]:
    """Synchronous local stub until NeMo / torch training is wired."""
    print(f"[training] local job={job_name}")
    print("[training] TODO: wire NeMo FastConformer phoneme recipe")
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}")
    return {"status": "local_stub", "job_name": job_name}


@app.function(image=image, timeout=600)
def train_remote(job_name: str) -> dict[str, Any]:
    """Remote smoke: structured logs + JSON return; no model weights written."""
    print(f"[training] remote job={job_name}", flush=True)
    print("[training] stub: replace with NeMo FastConformer phoneme training", flush=True)
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}", flush=True)
    return {"status": "completed", "job_name": job_name}


@app.local_entrypoint()
def modal_main(job_name: str = "fastconformer-phoneme-smoke") -> None:
    train_remote.remote(job_name)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch standalone lab training job (local)")
    p.add_argument("--job-name", default="fastconformer-phoneme-smoke")
    args = p.parse_args()
    train(args.job_name)


if __name__ == "__main__":
    main()
