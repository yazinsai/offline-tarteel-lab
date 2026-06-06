"""Hugging Face discovery: list models/datasets, score lightly, emit JSON backlog."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
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
    likes_score = (item.likes or 0) ** 0.5
    tag_bonus = 0.0
    for t in item.tags:
        tl = t.lower()
        if tl in ("arabic", "ar", "quran", "islamic", "phoneme", "ctc", "asr", "speech"):
            tag_bonus += 2.0
    return d + likes_score + tag_bonus


def main() -> None:
    p = argparse.ArgumentParser(description="HF intake backlog")
    p.add_argument("--query", default="arabic speech asr", help="Search string")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--kind", choices=("model", "dataset", "both"), default="model")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write ranked JSON to this file (still prints to stdout unless --quiet)",
    )
    p.add_argument(
        "--append-to",
        type=Path,
        default=None,
        help="Merge with existing backlog JSON array (dedupe by id, keep higher score)",
    )
    p.add_argument("--quiet", action="store_true", help="Do not print JSON to stdout")
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

    if args.append_to and args.append_to.is_file():
        try:
            prev = json.loads(args.append_to.read_text(encoding="utf-8"))
            if isinstance(prev, list):
                by_id: dict[str, dict[str, Any]] = {}
                for row in prev:
                    if isinstance(row, dict) and "id" in row:
                        by_id[str(row["id"])] = row
                for row in ranked:
                    rid = str(row["id"])
                    old = by_id.get(rid)
                    if old is None or float(row.get("score") or 0) >= float(
                        old.get("score") or 0,
                    ):
                        by_id[rid] = row
                ranked = sorted(by_id.values(), key=lambda z: float(z.get("score") or 0), reverse=True)
        except json.JSONDecodeError as e:
            print(f"Warning: could not merge --append-to: {e}", file=sys.stderr)

    text = json.dumps(ranked, indent=2)
    if not args.quiet:
        print(text)
        print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
