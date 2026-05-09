"""Standalone Modal training entrypoint for lab phoneme smoke jobs.

Keeps a real Modal local entrypoint so `modal run path/to/this/file.py` works
(the autopilot uses that form). Local smoke: `python ... --job-name foo`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import modal

app = modal.App("offline-tarteel-fastconformer-phoneme-lab")


def train(job_name: str) -> None:
    # Keep this local entrypoint runnable even before real training code is added.
    print(f"[training] starting job={job_name}")
    print("[training] TODO: wire NeMo/torch training loop in this file")
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}")


@app.local_entrypoint()
def run(job_name: str = "fastconformer-phoneme-smoke") -> None:
    train(job_name)


def main() -> None:
    p = argparse.ArgumentParser(description="Launch standalone lab training job (local Python)")
    p.add_argument("--job-name", default="fastconformer-phoneme-smoke")
    args = p.parse_args()
    train(args.job_name)


if __name__ == "__main__":
    main()
