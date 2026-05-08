# offline-tarteel-lab

Autonomous experimentation for **offline streaming Quran recitation** (verse tracking + correction).

This repo is the **research velocity** side of the split:

- Corpus QA gate + immutable dataset registry
- Tiered evaluation (fast ONNX loop → full corpus → browser/RN truth)
- Hugging Face discovery intake (models, datasets, papers)
- Cursor SDK cloud planner / workers / judge loop
- Modal training/eval integration hooks

**Consumer-facing packages** live in [`offline-tarteel-sdk`](https://github.com/yazinsai/offline-tarteel-sdk) (standalone repo).

## Quick start

```bash
cd offline-tarteel-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Fail if corpora are inconsistent (run from monorepo root containing benchmark/)
python -m lab_tools.validate_corpus

# List discovered experiments (manifest-driven)
python -m lab_tools.discover_experiments --experiments-dir ../experiments

# HF intake (requires HF_TOKEN optional)
python -m lab_tools.hf_intake --query "arabic asr" --limit 20

# Write a promotion record for the SDK repo (after judge accepts)
python -m lab_tools.promote --run-id RUN123 --output ../offline-tarteel-sdk/releases/promotions/
```

## Environment

| Variable | Meaning |
|----------|---------|
| `OFFLINE_TARTEEL_ROOT` | Root of the reference `offline-tarteel` tree (default: parent of this directory). |
| `HF_TOKEN` | Hugging Face API token for higher rate limits (optional). |
| `CURSOR_API_KEY` | For orchestration workers using `@cursor/sdk`. |

## Layout

See [docs/LAYOUT.md](docs/LAYOUT.md).
