# Modal (lab)

GPU training and heavy eval stay in the reference tree under `scripts/*_modal.py`.

From lab, invoke via:

```bash
cd ../../   # offline-tarteel root
modal run --detach scripts/train_fastconformer_phoneme_modal.py
```

Record each Modal run id + volume path in the lab run record (`commands[]` + `artifact_hashes`).
