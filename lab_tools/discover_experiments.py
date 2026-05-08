"""Auto-discover experiments: each dir under experiments/ with experiment.yaml + run.py."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def has_run_py(d: Path) -> bool:
    return (d / "run.py").is_file()


def load_run_module(run_py: Path):
    spec = importlib.util.spec_from_file_location(f"exp_{run_py.parent.name}", run_py)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description="Discover experiments from directory")
    p.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Default: OFFLINE_TARTEEL_ROOT/experiments",
    )
    p.add_argument("--json", action="store_true", help="Print JSON list")
    args = p.parse_args()
    from lab_tools.paths import reference_root

    exp_root = args.experiments_dir or (reference_root() / "experiments")
    exp_root = exp_root.resolve()
    if not exp_root.is_dir():
        print(f"No experiments dir: {exp_root}", file=sys.stderr)
        sys.exit(1)

    out: list[dict[str, Any]] = []
    for child in sorted(exp_root.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or child.name == "templates":
            continue
        if not has_run_py(child):
            continue
        meta_path = child / "experiment.yaml"
        meta = load_yaml(meta_path) if meta_path.is_file() else {}
        name = child.name
        mod = None
        load_error: str | None = None
        try:
            mod = load_run_module(child / "run.py")
        except Exception as e:
            load_error = str(e)
        multi = bool(mod and hasattr(mod, "list_models"))
        entry: dict[str, Any] = {
            "name": name,
            "path": str(child),
            "manifest": str(meta_path) if meta_path.is_file() else None,
            "kind": meta.get("kind", "model_only"),
            "hypothesis": meta.get("hypothesis", ""),
            "multi_model": multi,
        }
        if load_error:
            entry["load_error"] = load_error
        if multi and mod is not None:
            try:
                models = mod.list_models()
                entry["models"] = list(models)
            except Exception as e:
                entry["models_error"] = str(e)
        out.append(entry)

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for e in out:
            mm = f" [{', '.join(e.get('models', []))}]" if e.get("models") else ""
            print(f"- {e['name']} ({e['kind']}){mm}")


if __name__ == "__main__":
    main()
