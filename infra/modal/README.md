# Modal (lab)

GPU training and heavy eval entrypoints should live in this repo under `training/`.

From lab, invoke via:

```bash
modal run --detach training/train_fastconformer_phoneme_modal.py
```

Record each Modal run id + volume path in the lab run record (`commands[]` + `artifact_hashes`).
