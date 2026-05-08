"""Standalone Modal training entrypoint placeholder for lab experiments.

Replace the train() body with your real NeMo training logic.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone


def train(job_name: str) -> None:
    # Keep this local entrypoint runnable even before real training code is added.
    print(f"[training] starting job={job_name}")
    print("[training] TODO: wire NeMo/torch training loop in this file")
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Launch standalone lab training job")
    p.add_argument("--job-name", default="fastconformer-phoneme-smoke")
    args = p.parse_args()
    train(args.job_name)


if __name__ == "__main__":
    main()
