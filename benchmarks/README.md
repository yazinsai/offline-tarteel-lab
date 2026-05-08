# Benchmarks (lab)

- **`schemas/run_record.schema.json`** — canonical JSON Schema for experiment run provenance.
- **Tiers**
  - **Tier 1:** `python -m lab_tools.eval_tier --tier 1` (corpus QA + fast gates)
  - **Tier 2:** `python -m lab_tools.eval_tier --tier 2 --experiment <name>` (full Python benchmark)
  - **Tier 3:** `python -m lab_tools.eval_tier --tier 3` (browser/frontend stability harness)

Promotion to the SDK repo requires Tier 3 evidence attached to a run record.
