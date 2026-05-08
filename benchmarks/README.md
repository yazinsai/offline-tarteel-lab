# Benchmarks (lab)

- **`schemas/run_record.schema.json`** — canonical JSON Schema for experiment run provenance.
- **Tiers**
  - **Tier 1:** `python -m lab_tools.eval_tier --tier 1` (corpus QA + optional ffmpeg decode smoke)
  - **Tier 2:** `python -m lab_tools.eval_tier --tier 2 --experiment <name>` (local experiment benchmark run)
  - **Tier 3:** `python -m lab_tools.eval_tier --tier 3` (promotion gate from local Tier-2 evidence)

Promotion requires Tier 3 evidence attached to a run record.
