"""Hugging Face discovery: list models/datasets, score lightly, emit JSON backlog."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from typing import Any

from huggingface_hub import HfApi


@dataclass
class HFItem:
    id: str
    kind: str
    downloads: int | None
    likes: int | None
    pipeline_tag: str | None
    license: str | None
    tags: list[str]


def score_item(item: HFItem) -> float:
    """Heuristic ranking; planner can replace with learned model."""
    d = (item.downloads or 0) ** 0.25
    l = (item.likes or 0) ** 0.5
    tag_bonus = 0.0
    for t in item.tags:
        tl = t.lower()
        if tl in ("arabic", "ar", "quran", "islamic", "phoneme", "ctc", "asr", "speech"):
            tag_bonus += 2.0
    return d + l + tag_bonus


def main() -> None:
    p = argparse.ArgumentParser(description="HF intake backlog")
    p.add_argument("--query", default="arabic speech asr", help="Search string")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--kind", choices=("model", "dataset", "both"), default="model")
    args = p.parse_args()
    api = HfApi()

    items: list[HFItem] = []

    if args.kind in ("model", "both"):
        for m in api.list_models(search=args.query, limit=args.limit):
            lic = None
            if getattr(m, "card_data", None) and m.card_data:
                lic = getattr(m.card_data, "license", None)
            if isinstance(lic, list):
                lic = lic[0] if lic else None
            items.append(
                HFItem(
                    id=m.modelId,
                    kind="model",
                    downloads=getattr(m, "downloads", None),
                    likes=getattr(m, "likes", None),
                    pipeline_tag=getattr(m, "pipeline_tag", None),
                    license=str(lic) if lic else None,
                    tags=list(getattr(m, "tags", None) or []),
                ),
            )

    if args.kind in ("dataset", "both"):
        try:
            for d in api.list_datasets(search=args.query, limit=args.limit):
                lic = None
                try:
                    cd = getattr(d, "card_data", None)
                    if cd and getattr(cd, "license", None):
                        lic = str(cd.license)
                except Exception:
                    lic = None
                items.append(
                    HFItem(
                        id=d.id,
                        kind="dataset",
                        downloads=getattr(d, "downloads", None),
                        likes=getattr(d, "likes", None),
                        pipeline_tag=None,
                        license=lic,
                        tags=list(getattr(d, "tags", None) or []),
                    ),
                )
        except Exception as e:
            print(f"Warning: dataset search failed: {e}", file=sys.stderr)

    ranked: list[dict[str, Any]] = []
    for it in sorted(items, key=score_item, reverse=True):
        ranked.append({**asdict(it), "score": score_item(it)})

    json.dump(ranked, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
