# Training (standalone)

This directory contains lab-owned training entrypoints.

## Run locally

```bash
python -m training.train_fastconformer_phoneme_modal --job-name quick-smoke
```

## Run on Modal

```bash
modal run --detach training/train_fastconformer_phoneme_modal.py --job-name quick-smoke
```
