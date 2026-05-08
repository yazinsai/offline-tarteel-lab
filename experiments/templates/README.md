# Experiment templates

1. Copy `experiment.yaml` into `experiments/<slug>/` beside `run.py`.
2. `kind` must match plan taxonomy: `model_only`, `runtime_only`, or `joint_model_runtime`.
3. Discover without editing `benchmark/runner.py`:

```bash
python -m lab_tools.discover_experiments --experiments-dir ../../experiments --json
```

(Migrate `benchmark/runner.py` to auto-discovery in the reference repo when ready.)
