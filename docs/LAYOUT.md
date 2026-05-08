# Repository layout

```
offline-tarteel-lab/
  datasets/           # Registry + checksums for benchmark corpora
  benchmarks/       # Schemas, tier docs, comparators
  orchestration/    # Cursor SDK planner/worker/judge (TypeScript)
  experiments/      # Templates + manifest convention
  infra/modal/      # Notes and wrappers pointing at reference scripts
  artifacts/        # Run metadata (gitignored large blobs); manifests committed
  lab_tools/        # Python CLI package (validate, discover, eval, hf, promote)
  docs/
```

## Handoff to SDK

Promotion manifests are JSON files written under `offline-tarteel-sdk/releases/promotions/`.
The SDK release pipeline should only ship artifacts referenced in an accepted promotion record.
