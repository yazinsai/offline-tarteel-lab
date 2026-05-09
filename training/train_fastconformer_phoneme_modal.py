"""Standalone Modal training entrypoint placeholder for lab experiments.

Kept minimal so `python -m modal run ...` resolves a local entrypoint; extend train()
with NeMo/torch when wiring real phoneme training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import modal

app = modal.App("offline-tarteel-fastconformer-phoneme-smoke")


def train(job_name: str) -> None:
    print(f"[training] starting job={job_name}")
    print("[training] TODO: wire NeMo/torch training loop in this file")
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}")


@app.local_entrypoint()
def lab_modal_train(job_name: str = "fastconformer-phoneme-smoke") -> None:
    train(job_name)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch standalone lab training job")
    p.add_argument("--job-name", default="fastconformer-phoneme-smoke")
    args = p.parse_args()
    train(args.job_name)


if __name__ == "__main__":
    main()
