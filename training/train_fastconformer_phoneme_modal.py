"""Standalone Modal training entrypoint for lab experiments.

`modal run --detach training/train_fastconformer_phoneme_modal.py --job-name ...`
invokes the local entrypoint below. For a no-Modal smoke: `python .../train_fastconformer_phoneme_modal.py`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import modal

app = modal.App("offline-tarteel-lab-phoneme-smoke")


def train(job_name: str) -> None:
    print(f"[training] starting job={job_name}")
    print("[training] TODO: wire NeMo/torch training loop in this file")
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}")


@app.local_entrypoint()
def main(job_name: str = "fastconformer-phoneme-smoke") -> None:
    train(job_name)


def _legacy_cli() -> None:
    p = argparse.ArgumentParser(description="Launch standalone lab training job (local Python)")
    p.add_argument("--job-name", default="fastconformer-phoneme-smoke")
    args = p.parse_args()
    train(args.job_name)


if __name__ == "__main__":
    _legacy_cli()
