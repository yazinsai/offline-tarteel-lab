"""Modal training entrypoint for FastConformer phoneme lab jobs.

Defines a Modal :func:`local_entrypoint` so ``modal run`` succeeds; the same
module stays runnable via ``python3 training/...`` when Modal is unavailable.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import modal

app = modal.App("offline-tarteel-phoneme-lab")


def train(job_name: str) -> None:
    print(f"[training] starting job={job_name}")
    print("[training] TODO: wire NeMo/torch training loop in this file")
    print(f"[training] timestamp={datetime.now(timezone.utc).isoformat()}")


@app.local_entrypoint()
def main(job_name: str = "fastconformer-phoneme-smoke") -> None:
    train(job_name)


def main_cli() -> None:
    p = argparse.ArgumentParser(description="Launch standalone lab training job")
    p.add_argument("--job-name", default="fastconformer-phoneme-smoke")
    args = p.parse_args()
    train(args.job_name)


if __name__ == "__main__":
    main_cli()
