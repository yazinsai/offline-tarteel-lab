"""Standalone Modal training entrypoint placeholder for lab experiments.

Replace the train() body with your real NeMo training logic.

`modal run` dispatches via the Modal app local entrypoint; `python -m` invokes the
CLI path directly without importing Modal decorators at module level conflicts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import modal

app = modal.App("offline-tarteel-fastconformer-phoneme-smoke")


def train(job_name: str) -> None:
    # Keep this local entrypoint runnable even before real training code is added.
    print(f"[training] starting job={job_name}")
    print("[training] TODO: wire NeMo/torch training loop in this file")
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}")


@app.local_entrypoint()
def autopilot_training(job_name: str = "fastconformer-phoneme-autopilot") -> None:
    """Invoked by `modal run [...]training/train_fastconformer_phoneme_modal.py`."""
    train(job_name)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch standalone lab training job")
    p.add_argument("--job-name", default="fastconformer-phoneme-smoke")
    args = p.parse_args()
    train(args.job_name)


if __name__ == "__main__":
    main()
