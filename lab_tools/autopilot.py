"""Autopilot task generator for unattended lab progress."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from lab_tools.experiment_ledger import champion, failed_families, read_entries, worst_slice
from lab_tools.task_queue import (
    add_task_once,
    count_active,
    load_state,
    queue_lock,
    save_state,
    state_path,
)

SMOKE_RUNTIME_PLATEAU_THRESHOLD = 4
SMOKE_RUNTIME_PLATEAU_MAX_ALIGNMENT = 0.01
CHANGE_CLASS_FAILURE_THRESHOLD = 2
POPULATION_REFINE_LIMIT = 5
POPULATION_EXPLORATION_WEIGHT = 0.14
POPULATION_DIVERSITY_WEIGHT = 0.04


@dataclass(frozen=True)
class Candidate:
    key: str
    kind: str
    title: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class PopulationPick:
    entry: dict[str, Any]
    exploitation: float
    exploration: float
    diversity: float
    score: float
    visits: int
    mutation_visits: int


def _key_token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))


def _infer_change_class(text: str) -> str | None:
    text = text.lower()
    patterns = [
        ("ctc_beam_or_decode_knob", r"\b(beam|temp|temperature|length[ -]?norm|lennorm|sharpen|hyp[-_ ]?order|hypothesis ordering)\b"),
        ("reference_ctc_fusion", r"\b(reference|v4|tlog).*(ctc|fusion|discovery)|\bctc[-_ ]?fusion\b"),
        ("matcher_shortlist_widen", r"\b(shortlist|retrieval head|top_k|top surah|max_span|span search|rerank)\b"),
        ("queue_order_block", r"\bwrong[_ -]?task[_ -]?ordering|fifo|queue[-_ ]?order\b"),
        ("smoke_runtime", r"\bsmoke|runtime\.adaptive|threshold|chunk|overlap|hysteresis|debounce\b"),
    ]
    for change_class, pattern in patterns:
        if re.search(pattern, text):
            return change_class
    return None


def change_class_for_payload(payload: dict[str, Any] | None, *, fallback_key: str = "") -> str | None:
    payload = payload or {}
    explicit = payload.get("change_class")
    if explicit:
        return _key_token(explicit)
    fields = [
        fallback_key,
        payload.get("autopilot_key"),
        payload.get("preflight_note"),
        payload.get("task_note"),
        payload.get("candidate_experiment"),
        payload.get("experiment_probe"),
        payload.get("experiment"),
        payload.get("param"),
        payload.get("agent_instructions"),
    ]
    return _infer_change_class(" ".join(str(v or "") for v in fields))


def _entry_change_class(entry: dict[str, Any]) -> str | None:
    params = entry.get("parameters") or {}
    return change_class_for_payload(params, fallback_key=str(entry.get("experiment_family") or ""))


def failed_change_classes(
    entries: list[dict[str, Any]] | None = None,
    *,
    threshold: int = CHANGE_CLASS_FAILURE_THRESHOLD,
) -> set[str]:
    failed_families_by_class: dict[str, set[str]] = {}
    for entry in entries if entries is not None else read_entries():
        if entry.get("status") not in {"rejected", "failed", "superseded"}:
            continue
        change_class = _entry_change_class(entry)
        if change_class:
            family = str(entry.get("experiment_family") or entry.get("run_id") or "")
            failed_families_by_class.setdefault(change_class, set()).add(family)
    return {
        change_class
        for change_class, families in failed_families_by_class.items()
        if len(families) >= threshold
    }


def _promoted_keys(entries: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for entry in entries:
        if entry.get("status") not in {"promoted", "accepted", "merged"}:
            continue
        params = entry.get("parameters") or {}
        key = params.get("autopilot_key") or entry.get("experiment_family")
        if key:
            keys.add(str(key))
    return keys


def _missing_dependencies(candidate: Candidate, promoted_keys: set[str]) -> list[str]:
    deps = candidate.payload.get("dependency_autopilot_keys") or []
    return [str(dep) for dep in deps if str(dep) not in promoted_keys]


def candidates(entries: list[dict[str, Any]] | None = None) -> list[Candidate]:
    entries = entries if entries is not None else read_entries()
    static = [
        Candidate(
            key="runtime.smoke_streaming_baseline",
            kind="runtime_only",
            title="Improve streaming first-verse baseline",
            payload={
                "experiment": "smoke",
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Inspect the current experiments and benchmark manifests. Create or update a "
                    "small runnable experiment under experiments/ that improves first-verse "
                    "streaming accuracy without adding large artifacts. Keep it deterministic and "
                    "compatible with lab_tools.tier2_local predict(audio_path)."
                ),
            },
        ),
        Candidate(
            key="runtime.threshold_sweep.first_match",
            kind="runtime_only",
            title="Tune first-match streaming threshold",
            payload={
                "param": "FIRST_MATCH_THRESHOLD",
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Search for runtime threshold logic or create an experiment that models a "
                    "first-match threshold sweep. Evaluate multiple candidates locally and commit "
                    "the best deterministic experiment only if it improves Tier-2 accuracy."
                ),
            },
        ),
        Candidate(
            key="runtime.chunk_window_sweep",
            kind="runtime_only",
            title="Tune streaming chunk/window size",
            payload={
                "param": "chunk_seconds",
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Explore chunk/window parameters for streaming recitation tracking. Prefer "
                    "small code changes and record the selected parameter vector in the task "
                    "payload or experiment metadata."
                ),
            },
        ),
        Candidate(
            key="model.fastconformer_phoneme_smoke",
            kind="model_only",
            title="Evaluate phoneme FastConformer training candidate",
            payload={
                "modal_training": True,
                "job_name": "fastconformer-phoneme-autopilot",
                "min_accuracy": 0.85,
                "max_onnx_mb": 200,
                "agent_instructions": (
                    "Only launch Modal when LAB_AUTONOMY_ALLOW_MODAL is enabled. Otherwise inspect "
                    "the training entrypoint and prepare a small, reviewable training/eval "
                    "improvement that keeps local tests passing. Never commit model binaries."
                ),
            },
        ),
        Candidate(
            key="joint.model_runtime_export_contract",
            kind="joint_model_runtime",
            title="Tighten ONNX export and streaming wrapper contract",
            payload={
                "min_accuracy": 0.85,
                "max_onnx_mb": 200,
                "agent_instructions": (
                    "Improve the path from a candidate model artifact to an offline streaming "
                    "consumer wrapper. Add tests or metadata checks for ONNX size/hash and package "
                    "compatibility. Do not commit generated model artifacts."
                ),
            },
        ),
    ]
    planned = population_guided_candidates(entries) + ledger_guided_candidates(entries) + static + adaptive_runtime_candidates()
    if smoke_runtime_plateau(entries):
        planned = non_smoke_escalation_candidates(entries) + [
            c for c in planned if not _candidate_blocked_by_smoke_runtime_plateau(c)
        ]
    return planned


def _is_smoke_runtime_plateau_entry(entry: dict[str, Any]) -> bool:
    params = entry.get("parameters") or {}
    components = entry.get("components") or {}
    try:
        alignment = float(components.get("streaming_alignment_accuracy") or 0.0)
    except (TypeError, ValueError):
        alignment = 0.0
    return (
        entry.get("status") == "rejected"
        and entry.get("experiment_kind") == "runtime_only"
        and entry.get("corpus_revision") == "test_corpus_v3"
        and params.get("experiment") == "smoke"
        and params.get("full_corpus_gate") is True
        and "min_accuracy_not_met" in set(entry.get("failure_modes") or [])
        and alignment <= SMOKE_RUNTIME_PLATEAU_MAX_ALIGNMENT
    )


def smoke_runtime_plateau(
    entries: list[dict[str, Any]] | None = None,
    *,
    threshold: int = SMOKE_RUNTIME_PLATEAU_THRESHOLD,
) -> bool:
    entries = entries if entries is not None else read_entries()
    return sum(1 for entry in entries if _is_smoke_runtime_plateau_entry(entry)) >= threshold


def _plateau_attempt_generation(entries: list[dict[str, Any]]) -> int:
    attempts = 0
    for entry in entries:
        family = str(entry.get("experiment_family") or "")
        params = entry.get("parameters") or {}
        if _is_smoke_runtime_plateau_entry(entry):
            attempts += 1
        elif params.get("blocked_family") == "smoke_runtime_plateau":
            attempts += 1
        elif family.startswith(("baseline.reference_shipped_fastconformer_v4_tlog.", "escalate.non_smoke.")):
            attempts += 1
    return attempts


def _candidate_blocked_by_smoke_runtime_plateau(candidate: Candidate) -> bool:
    key = candidate.key
    experiment = candidate.payload.get("experiment")
    return candidate.kind == "runtime_only" and (
        experiment == "smoke" or key.startswith("runtime.")
    )


def non_smoke_escalation_candidates(entries: list[dict[str, Any]] | None = None) -> list[Candidate]:
    entries = entries if entries is not None else read_entries()
    failed_count = sum(1 for entry in entries if _is_smoke_runtime_plateau_entry(entry))
    generation = _plateau_attempt_generation(entries)
    baseline_key = f"baseline.reference_shipped_fastconformer_v4_tlog.{generation:02d}"
    first_model_key = f"escalate.non_smoke.model_only.{generation:02d}.01"
    baseline = Candidate(
        key=baseline_key,
        kind="joint_model_runtime",
        title="Port shipped fastconformer-phoneme v4-tlog baseline from reference repo",
        payload={
            "blocked_family": "smoke_runtime_plateau",
            "change_class": "reference_port",
            "plateau_failures": failed_count,
            "reference_repo": "../offline-tarteel",
            "reference_repo_url": "https://github.com/yazinsai/offline-tarteel.git",
            "reference_baseline": "fastconformer-phoneme v4-tlog browser/RN streaming",
            "target_corpus": "test_corpus_v3",
            "target_correct_range": [223, 225],
            "target_recall": 0.893,
            "min_accuracy": 0.8,
            "agent_instructions": (
                "Stop using the smoke baseline. Port or wrap the shipped baseline from "
                "../offline-tarteel if present; otherwise clone "
                "https://github.com/yazinsai/offline-tarteel.git into /tmp/offline-tarteel-reference. "
                "Use fastconformer-phoneme v4-tlog plus the browser/RN RecitationTracker "
                "and QuranDB matching behavior. Use the reference files "
                "web/frontend/src/lib/{tracker.ts,quran-db.ts,phoneme-trie.ts,normalizer.ts,"
                "levenshtein.ts,ctc-rescore.ts,types.ts} and data/quran_phonemes.json as the "
                "source of truth. The known v3 baseline is 223-225/256 correct; reproduce that "
                "from audio/model/tracker behavior before optimizing. Do not read predictions "
                "from stability JSON, filenames, manifests, row order, or sidecar labels."
            ),
        },
    )
    focus = [
        (
            "model_only",
            "Evaluate non-smoke ASR candidate on full corpus",
            "asr_model_evidence",
            "Build or wire a real non-smoke ASR experiment under experiments/ that predicts from audio/model output. Do not tune smoke runtime parameters. Do not submit beam/temp/length-norm/logprob-sharpen/hypothesis-ordering-only changes unless they add new acoustic/model evidence. If Modal is unavailable, prepare deterministic local evaluation scaffolding and reject honestly rather than emitting another smoke-runtime probe.",
        ),
        (
            "joint_model_runtime",
            "Improve real ASR-to-Quran matcher path",
            "joint_asr_matcher_new_evidence",
            "Work on a non-smoke matcher/ASR integration that uses transcript or model evidence from audio. Avoid filename/path/manifest labels and avoid smoke runtime knobs. Do not submit shortlist/rerank/beam-only probes unless a promoted ASR/reference dependency changed the input evidence. Keep committed changes inside experiments/, tests/, benchmark/results/, or small JSON metadata.",
        ),
        (
            "model_only",
            "Probe lightweight offline recognition baseline",
            "lightweight_audio_baseline",
            "Create a small reviewable experiment baseline that gets surah/ayah from actual audio-derived recognition signals, not corpus metadata. Prefer existing shared audio/Quran utilities and commit no model binaries.",
        ),
        (
            "joint_model_runtime",
            "Add non-smoke failure analysis experiment",
            "failure_analysis_new_signal",
            "Use the full v3 failure pattern to create a targeted non-smoke experiment or matcher harness. The goal is to escape the 1/256 smoke plateau, not to adjust streaming metadata.",
        ),
    ]
    out: list[Candidate] = [baseline]
    for i in range(1, 13):
        kind, title, change_class, instructions = focus[(i - 1) % len(focus)]
        payload: dict[str, Any] = {
            "blocked_family": "smoke_runtime_plateau",
            "change_class": change_class,
            "plateau_failures": failed_count,
            "min_accuracy": 0.8,
            "agent_instructions": instructions,
        }
        if kind == "joint_model_runtime":
            payload["dependency_autopilot_keys"] = [baseline_key, first_model_key]
        out.append(
            Candidate(
                key=f"escalate.non_smoke.{kind}.{generation:02d}.{i:02d}",
                kind=kind,
                title=f"{title} variant {i:02d}",
                payload=payload,
            ),
        )
    return out


def ledger_guided_candidates(entries: list[dict[str, Any]] | None = None) -> list[Candidate]:
    entries = entries if entries is not None else read_entries()
    champ = champion(entries)
    weak = worst_slice(entries)
    out: list[Candidate] = []

    if weak and champ:
        slice_name, slice_data = weak
        run_token = _key_token(champ.get("run_id"))
        out.append(
            Candidate(
                key=f"runtime.repair_slice.{_key_token(slice_name)}.{run_token}",
                kind="runtime_only",
                title=f"Repair weak corpus slice: {slice_name}",
                payload={
                    "experiment": (champ.get("parameters") or {}).get("experiment", "smoke"),
                    "target_slice": slice_name,
                    "target_slice_score": slice_data.get("score"),
                    "champion_run_id": champ.get("run_id"),
                    "baseline_objective": champ.get("objective"),
                    "min_accuracy": 0.8,
                    "agent_instructions": (
                        f"Optimize corpus v3 slice '{slice_name}' without regressing the champion. "
                        "Inspect manifest tags and per-sample failures for this slice, make a small "
                        "runtime-only matcher/tracker change, then emit scorer-compatible slice metrics."
                    ),
                },
            ),
        )

    if champ:
        params = dict(champ.get("parameters") or {})
        family = _key_token(champ.get("experiment_family") or "champion")
        run_token = _key_token(champ.get("run_id"))
        out.append(
            Candidate(
                key=f"runtime.exploit_champion.{family}.{run_token}",
                kind="runtime_only",
                title="Exploit current champion with local runtime mutation",
                payload={
                    "experiment": params.get("experiment", "smoke"),
                    "champion_run_id": champ.get("run_id"),
                    "champion_parameters": params,
                    "baseline_objective": champ.get("objective"),
                    "min_accuracy": max(0.8, float(champ.get("objective") or 0.0)),
                    "agent_instructions": (
                        "Start from the champion parameter vector and try a narrow deterministic "
                        "runtime mutation. Only keep the change if the scorer objective improves beyond "
                        "variance and no critical slice regresses."
                    ),
                },
            ),
        )

    out.append(
        Candidate(
            key="runtime.explore_diverse.v3",
            kind="runtime_only",
            title="Explore a diverse corpus-v3 runtime strategy",
            payload={
                "experiment": "smoke",
                "exploration": True,
                "min_accuracy": 0.8,
                "agent_instructions": (
                    "Try one runtime-only strategy that is structurally different from the current "
                    "champion and recent rejected families. Keep changes small, deterministic, and "
                    "scored with the canonical objective."
                ),
            },
        ),
    )
    return out


def _population_score(entry: dict[str, Any]) -> float:
    population = entry.get("population") or {}
    try:
        return float(population.get("search_rating"))
    except (TypeError, ValueError):
        try:
            return float(entry.get("objective") or 0.0)
        except (TypeError, ValueError):
            return 0.0


def _population_mutation_type(entry: dict[str, Any]) -> str:
    population = entry.get("population") or {}
    params = entry.get("parameters") or {}
    return str(
        population.get("mutation_type")
        or params.get("mutation_type")
        or params.get("change_class")
        or entry.get("experiment_kind")
        or "unknown"
    )


def _population_visit_count(entry: dict[str, Any]) -> int:
    population = entry.get("population") or {}
    try:
        explicit = int(population.get("visit_count") or 0)
    except (TypeError, ValueError):
        explicit = 0
    return max(1, explicit + 1)


def _population_novelty_tags(entry: dict[str, Any]) -> set[str]:
    population = entry.get("population") or {}
    params = entry.get("parameters") or {}
    tags = set()
    for value in (
        population.get("novelty_tags"),
        params.get("novelty_tags"),
        params.get("focus_bucket"),
        params.get("target_slice"),
        params.get("change_class"),
    ):
        if isinstance(value, str):
            tags.add(value)
        elif isinstance(value, list | tuple | set):
            tags.update(str(item) for item in value if item is not None)
    return {tag for tag in tags if tag}


def population_policy_picks(entries: list[dict[str, Any]] | None = None) -> list[PopulationPick]:
    """Rank prior runs with a deterministic P-UCB-style exploration policy."""
    entries = entries if entries is not None else read_entries()
    eligible = [
        entry
        for entry in entries
        if isinstance(entry.get("population"), dict)
        and entry.get("run_id")
        and entry.get("objective") is not None
        and entry.get("status") in {"promoted", "accepted", "merged", "rejected"}
    ]
    if not eligible:
        return []

    mutation_counts = Counter(_population_mutation_type(entry) for entry in eligible)
    tag_counts: Counter[str] = Counter()
    for entry in eligible:
        tag_counts.update(_population_novelty_tags(entry))

    total_visits = sum(_population_visit_count(entry) for entry in eligible)
    picks: list[PopulationPick] = []
    for entry in eligible:
        visits = _population_visit_count(entry)
        mutation_type = _population_mutation_type(entry)
        mutation_visits = max(1, mutation_counts[mutation_type])
        exploitation = _population_score(entry)
        exploration = POPULATION_EXPLORATION_WEIGHT * math.sqrt(
            math.log(total_visits + 1.0) / mutation_visits
        )
        tags = _population_novelty_tags(entry)
        diversity = 0.0
        if tags:
            diversity = POPULATION_DIVERSITY_WEIGHT * sum(1.0 / tag_counts[tag] for tag in tags)
        score = exploitation + exploration + diversity
        picks.append(
            PopulationPick(
                entry=entry,
                exploitation=round(exploitation, 6),
                exploration=round(exploration, 6),
                diversity=round(diversity, 6),
                score=round(score, 6),
                visits=visits,
                mutation_visits=mutation_visits,
            ),
        )
    return sorted(
        picks,
        key=lambda pick: (
            pick.score,
            pick.exploitation,
            str(pick.entry.get("run_id") or ""),
        ),
        reverse=True,
    )


def population_guided_candidates(entries: list[dict[str, Any]] | None = None) -> list[Candidate]:
    out: list[Candidate] = []
    for rank, pick in enumerate(population_policy_picks(entries)[:POPULATION_REFINE_LIMIT], start=1):
        entry = pick.entry
        population = entry.get("population") or {}
        params = dict(entry.get("parameters") or {})
        family = _key_token(entry.get("experiment_family") or "population")
        run_token = _key_token(entry.get("run_id"))
        lineage_depth = int(population.get("lineage_depth") or 0) + 1
        mutation_type = str(population.get("mutation_type") or params.get("change_class") or "population_refine")
        novelty_tags = list(population.get("novelty_tags") or [])
        out.append(
            Candidate(
                key=f"population.refine.{family}.{run_token}",
                kind=str(entry.get("experiment_kind") or "joint_model_runtime"),
                title=f"Refine high-rated population candidate: {family}",
                payload={
                    "experiment": params.get("experiment"),
                    "selection_policy": "pucb_v1",
                    "selection_rank": rank,
                    "selection_score": pick.score,
                    "exploitation_score": pick.exploitation,
                    "exploration_bonus": pick.exploration,
                    "diversity_bonus": pick.diversity,
                    "population_visits": pick.visits,
                    "mutation_visits": pick.mutation_visits,
                    "parent_run_id": entry.get("run_id"),
                    "parent_objective": entry.get("objective"),
                    "parent_search_rating": population.get("search_rating"),
                    "lineage_depth": lineage_depth,
                    "mutation_type": mutation_type,
                    "novelty_tags": novelty_tags,
                    "min_accuracy": max(0.8, float(entry.get("objective") or 0.0)),
                    "agent_instructions": (
                        "Start from this parent run's parameter vector and diagnostics. Make one "
                        "bounded mutation selected by the population P-UCB policy. Prioritize "
                        "changes that can recover at least one current miss toward 95%+ full-corpus "
                        "accuracy: if the gold verse is absent from the candidate pool, change "
                        "candidate generation/span construction; if gold is present but low-rank, "
                        "change selector features or calibration. Preserve the validation contract, "
                        "record per-sample Tier-2 diagnostics, and reject honestly if the mandatory "
                        "champion preflight does not strictly improve."
                    ),
                },
            ),
        )
    return out


def adaptive_runtime_candidates() -> list[Candidate]:
    focuses = [
        ("chunk_seconds", "Tune streaming chunk duration"),
        ("overlap_seconds", "Tune streaming chunk overlap"),
        ("FIRST_MATCH_THRESHOLD", "Tune first-match threshold"),
        ("VERSE_MATCH_THRESHOLD", "Tune verse-match threshold"),
        ("smoothing_window", "Tune streaming smoothing window"),
        ("correction_hysteresis", "Tune correction hysteresis"),
        ("partial_match_margin", "Tune partial-match margin"),
        ("debounce_ms", "Tune correction debounce"),
    ]
    out: list[Candidate] = []
    for i in range(1, 25):
        param, title = focuses[(i - 1) % len(focuses)]
        out.append(
            Candidate(
                key=f"runtime.adaptive.{param}.{i:02d}",
                kind="runtime_only",
                title=f"{title} variant {i:02d}",
                payload={
                    "experiment": "smoke",
                    "param": param,
                    "change_class": "smoke_runtime",
                    "min_accuracy": 0.8,
                    "agent_instructions": (
                        "Create a small deterministic runtime experiment variant under "
                        "experiments/ with any supporting tests or benchmark JSON metadata. "
                        "Use artifacts/autonomy_failures as negative memory and avoid repeating "
                        "blocked paths. Do not touch training/, lab_tools/, orchestration/, "
                        ".github/, pyproject.toml, or generated model/audio artifacts."
                    ),
                },
            ),
        )
    return out


def _failure_records() -> list[dict[str, Any]]:
    root = state_path().parent.parent / "autonomy_failures"
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _task_ids_from_failure(record: dict[str, Any]) -> set[str]:
    task_ids: set[str] = set()
    for filename in record.get("changed_files", []):
        task_ids.update(re.findall(r"(task-[0-9a-f]{12})", str(filename)))
    return task_ids


def _retire_repeatedly_blocked_tasks(*, threshold: int = 2) -> list[str]:
    failures = _failure_records()
    if not failures:
        return []

    by_task: Counter[str] = Counter()
    blocked_non_runtime = 0
    for record in failures:
        changed = [str(p) for p in record.get("changed_files", [])]
        if any(
            p == "pyproject.toml"
            or p.startswith((".github/", "lab_tools/", "orchestration/", "training/"))
            for p in changed
        ):
            blocked_non_runtime += 1
        for task_id in _task_ids_from_failure(record):
            by_task[task_id] += 1

    retired: list[str] = []
    with queue_lock():
        state = load_state()
        changed = False
        for task in state.tasks:
            if task.status not in {"queued", "running", "needs_eval"}:
                continue
            payload = task.payload or {}
            repeated_task_failure = by_task[task.id] >= threshold
            plateau_escape = payload.get("blocked_family") == "smoke_runtime_plateau"
            unsafe_family = (
                task.kind in {"model_only", "joint_model_runtime"}
                and blocked_non_runtime >= threshold
                and not plateau_escape
            )
            if not (repeated_task_failure or unsafe_family):
                continue
            task.status = "rejected"
            task.judge_reasons = ["autopilot_failure_memory_retired"]
            task.notes = (
                "Retired by autopilot after repeated autonomous merge-gate failures. "
                "Future work should be split into runtime-only auto-merge tasks or normal "
                "human-review PRs for training/plumbing changes."
            )
            task.touch()
            retired.append(task.id)
            changed = True
        if changed:
            save_state(state)
    return retired


def _retire_smoke_runtime_plateau_tasks(entries: list[dict[str, Any]]) -> list[str]:
    if not smoke_runtime_plateau(entries):
        return []

    retired: list[str] = []
    with queue_lock():
        state = load_state()
        changed = False
        for task in state.tasks:
            if task.status not in {"queued", "running", "needs_eval"}:
                continue
            candidate = Candidate(
                key=str((task.payload or {}).get("autopilot_key") or ""),
                kind=task.kind,
                title=task.title,
                payload=task.payload or {},
            )
            payload = task.payload or {}
            stale_generic_escalation = (
                payload.get("blocked_family") == "smoke_runtime_plateau"
                and not payload.get("change_class")
                and not payload.get("reference_baseline")
                and str(payload.get("autopilot_key") or "").startswith("escalate.non_smoke.")
            )
            if not (
                _candidate_blocked_by_smoke_runtime_plateau(candidate) or stale_generic_escalation
            ):
                continue
            task.status = "rejected"
            task.judge_reasons = ["smoke_runtime_plateau_retired"]
            task.notes = (
                "Retired by autopilot after repeated honest full-corpus v3 smoke runtime "
                "failures at 1/256. Future tasks must start by porting the shipped "
                "fastconformer-phoneme v4-tlog baseline or use non-smoke ASR/model/matcher "
                "signals."
            )
            task.touch()
            retired.append(task.id)
            changed = True
        if changed:
            save_state(state)
    return retired


def plan(target_backlog: int) -> dict[str, Any]:
    entries = read_entries()
    retired = _retire_repeatedly_blocked_tasks()
    retired += _retire_smoke_runtime_plateau_tasks(entries)
    added: list[str] = []
    skipped: list[dict[str, Any]] = []
    blocked_families = failed_families(entries)
    blocked_change_classes = failed_change_classes(entries)
    promoted_keys = _promoted_keys(entries)
    champ = champion(entries)
    weak = worst_slice(entries)
    active = count_active()
    plateau = smoke_runtime_plateau(entries)
    policy_picks = population_policy_picks(entries)
    for candidate in candidates(entries):
        if candidate.key in blocked_families:
            skipped.append({"key": candidate.key, "reason": "blocked_family"})
            continue
        change_class = change_class_for_payload(candidate.payload, fallback_key=candidate.key)
        if change_class in blocked_change_classes:
            skipped.append(
                {
                    "key": candidate.key,
                    "reason": "blocked_change_class",
                    "change_class": change_class,
                },
            )
            continue
        missing = _missing_dependencies(candidate, promoted_keys)
        if missing:
            skipped.append(
                {
                    "key": candidate.key,
                    "reason": "missing_dependency",
                    "dependencies": missing,
                },
            )
            continue
        if active >= target_backlog:
            break
        task = add_task_once(
            candidate.kind,
            candidate.title,
            candidate.payload,
            key=candidate.key,
        )
        if task is not None:
            active += 1
            added.append(task.id)
    state = load_state()
    return {
        "active": active,
        "added": added,
        "retired": retired,
        "total_tasks": len(state.tasks),
        "queued": sum(1 for t in state.tasks if t.status == "queued"),
        "skipped": skipped,
        "champion": {
            "run_id": champ.get("run_id"),
            "objective": champ.get("objective"),
            "family": champ.get("experiment_family"),
        }
        if champ
        else None,
        "worst_slice": {"name": weak[0], "score": weak[1].get("score")} if weak else None,
        "blocked_families": sorted(blocked_families),
        "blocked_change_classes": sorted(blocked_change_classes),
        "smoke_runtime_plateau": plateau,
        "population_policy": [
            {
                "run_id": pick.entry.get("run_id"),
                "family": pick.entry.get("experiment_family"),
                "mutation_type": _population_mutation_type(pick.entry),
                "score": pick.score,
                "exploitation": pick.exploitation,
                "exploration": pick.exploration,
                "diversity": pick.diversity,
                "visits": pick.visits,
                "mutation_visits": pick.mutation_visits,
                "novelty_tags": sorted(_population_novelty_tags(pick.entry)),
            }
            for pick in policy_picks[:POPULATION_REFINE_LIMIT]
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Seed autonomous lab tasks when backlog is low")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan")
    sp.add_argument("--target-backlog", type=int, default=3)
    args = p.parse_args()

    if args.cmd == "plan":
        print(json.dumps(plan(args.target_backlog), indent=2))


if __name__ == "__main__":
    main()
