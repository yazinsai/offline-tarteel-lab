# Repository layout

```
offline-tarteel-lab/
  benchmark/          # Local corpora/manifests used by standalone gates
  datasets/           # Registry + checksums for benchmark corpora
  benchmarks/       # Schemas, tier docs, comparators
  orchestration/    # Cursor SDK planner/worker/judge (TypeScript)
  experiments/      # Templates + manifest convention
  infra/modal/      # Modal training/eval notes and local launch patterns
  artifacts/        # Run metadata (gitignored large blobs); manifests committed
  lab_tools/        # Python CLI package (validate, discover, eval, hf, promote)
  docs/
```

## Promotions

Promotion manifests are JSON files written by `lab_tools.promote`.
By default, write them under this repo (for example: `artifacts/promotions/`).
