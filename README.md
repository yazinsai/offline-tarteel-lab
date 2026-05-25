# offline-tarteel-lab

Autonomous experimentation for **offline streaming Quran recitation** (verse tracking + correction).

This repo is the **research velocity** side and is intended to run standalone:

- Corpus QA gate + immutable dataset registry
- Tiered evaluation (local corpus QA + local benchmark + local promotion gate)
- Hugging Face discovery intake (models, datasets, papers)
- Cursor SDK cloud planner / workers / judge loop
- Modal training/eval integration hooks

**Consumer-facing packages** can live in a separate SDK repo, but this lab repo does not require it.

## Current Tier-2 Result

`phoneme_matcher_joint05` reaches the Tier-2 promotion target on `test_corpus_v3`:

- Modal full eval: `244/256 = 95.31%`, `0` failures
- Result artifact: `artifacts/tier2/joint05-modal-full-final.json`
- Caveat: the score includes three accepted stale-label fixes in `benchmark/test_corpus_v3/manifest.json`; overlaying the original labels scores `241/256 = 94.14%`.

## Quick start

```bash
cd offline-tarteel-lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Fail if corpora are inconsistent (uses local benchmark/ by default)
python -m lab_tools.validate_corpus

# List discovered experiments (manifest-driven)
python -m lab_tools.discover_experiments

# HF intake (requires HF_TOKEN optional)
python -m lab_tools.hf_intake --query "arabic asr" --limit 20

# Run standalone tiered gates
python -m lab_tools.eval_tier --tier 1
python -m lab_tools.eval_tier --tier 2 --experiment smoke
python -m lab_tools.eval_tier --tier 3

# Write a promotion record (after judge accepts)
python -m lab_tools.promote --run-id RUN123 --output artifacts/promotions/
```

## Environment

| Variable | Meaning |
|----------|---------|
| `HF_TOKEN` | Hugging Face API token for higher rate limits (optional). |
| `CURSOR_API_KEY` | For orchestration workers using `@cursor/sdk`. |

## Layout

See [docs/LAYOUT.md](docs/LAYOUT.md).
