"""
Extend merged initial temporal rules for a fixed number of rounds.

Current scope:
  - Each round extends every current rule with one body atom drawn from the
    merged initial rule set.
  - The added atom must be new to the rule body.
  - Evaluation uses binding-aware evidence-set intersections.
  - Pruning is dispatched through a strategy interface.

Output layout:
    pipeline_output/driving_mini_extended_rules/
        extended_rules_manifest.json
        extended_rules_round_<n>.json
        extended_rules_round_<n>.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config
from src.exp_driving_videos.modules.extended_rules_pruning import (
    apply_pruning_strategies,
    normalize_strategy_names,
)


_EXTENDED_RULES_VERSION = 2


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "driving_mini_extended_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_prune_strategies(cfg: Dict[str, Any]) -> List[str]:
    raw_list = cfg.get("prune_strategies")
    if isinstance(raw_list, list):
        strategies = normalize_strategy_names([str(item) for item in raw_list])
        if strategies:
            return strategies
    return normalize_strategy_names([str(cfg.get("prune_strategy", "empty_evidence"))])


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "num_rounds": int(cfg.get("num_rounds", 3)),
        "evaluation_strategy": str(cfg.get("evaluation_strategy", "binding_aware_intersection")),
        "prune_strategies": _resolve_prune_strategies(cfg),
        "min_positive_support_to_extend": int(cfg.get("min_positive_support_to_extend", 1)),
        "same_confidence_smaller_evidence_enabled": bool(
            cfg.get("same_confidence_smaller_evidence_enabled", True)
        ),
    }


def _strip_trailing_dot(atom_text: str) -> str:
    return str(atom_text).strip().rstrip(".").strip()


def _normalize_atom_template(atom_text: str) -> str:
    normalized = _strip_trailing_dot(atom_text)
    return f"{normalized}." if normalized else ""


def _canonical_body_atoms(body_atoms: Iterable[str]) -> Tuple[str, ...]:
    normalized = {
        _normalize_atom_template(atom_text)
        for atom_text in body_atoms
        if _normalize_atom_template(atom_text)
    }
    return tuple(sorted(normalized))


def _build_clause(head_atom_template: str, body_atom_templates: Sequence[str]) -> str:
    head = _strip_trailing_dot(head_atom_template)
    body = ", ".join(_strip_trailing_dot(atom) for atom in body_atom_templates if _strip_trailing_dot(atom))
    if not head:
        return ""
    if not body:
        return f"{head}."
    return f"{head} :- {body}."


def _get_rule_body_atom_templates(rule: Dict[str, Any]) -> Tuple[str, ...]:
    body_atoms = rule.get("body_atom_templates")
    if isinstance(body_atoms, list):
        return _canonical_body_atoms(body_atoms)

    body_atom_template = str(rule.get("body_atom_template", "")).strip()
    if body_atom_template:
        return _canonical_body_atoms([body_atom_template])
    return ()


def _group_evidence_by_example(
    evidence_entries: Sequence[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in evidence_entries:
        example_id = str(entry.get("example_id", ""))
        grouped.setdefault(example_id, []).append(entry)
    return grouped


def _bindings_compatible(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> bool:
    for key, value in left.items():
        if key in right and str(right[key]) != str(value):
            return False
    return True


def _merge_bindings(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in left.items()}
    for key, value in right.items():
        merged[str(key)] = str(value)
    return merged


def _merge_matched_atoms(
    left: Dict[str, Any],
    right: Dict[str, Any],
) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in left.items()}
    for key, value in right.items():
        merged[str(key)] = str(value)
    return merged


def _dedupe_evidence_entries(
    evidence_entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, Tuple[Tuple[str, str], ...], Tuple[Tuple[str, str], ...]]] = set()
    for entry in evidence_entries:
        bindings = tuple(sorted((str(k), str(v)) for k, v in dict(entry.get("bindings", {})).items()))
        matched_atoms = tuple(
            sorted((str(k), str(v)) for k, v in dict(entry.get("matched_atoms", {})).items())
        )
        key = (str(entry.get("example_id", "")), bindings, matched_atoms)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _intersect_evidence_sets(
    parent_evidence_set: Sequence[Dict[str, Any]],
    added_atom_evidence_set: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    parent_by_example = _group_evidence_by_example(parent_evidence_set)
    added_by_example = _group_evidence_by_example(added_atom_evidence_set)
    shared_example_ids = sorted(set(parent_by_example) & set(added_by_example))

    intersected: List[Dict[str, Any]] = []
    for example_id in shared_example_ids:
        parent_entries = parent_by_example.get(example_id, [])
        added_entries = added_by_example.get(example_id, [])
        for parent_entry in parent_entries:
            parent_bindings = dict(parent_entry.get("bindings", {}))
            parent_matched_atoms = dict(parent_entry.get("matched_atoms", {}))
            for added_entry in added_entries:
                added_bindings = dict(added_entry.get("bindings", {}))
                if not _bindings_compatible(parent_bindings, added_bindings):
                    continue

                matched_atoms = _merge_matched_atoms(
                    parent_matched_atoms,
                    dict(added_entry.get("matched_atoms", {})),
                )
                intersected.append(
                    {
                        "video_id": str(parent_entry.get("video_id", added_entry.get("video_id", ""))),
                        "example_id": example_id,
                        "current_segment_id": str(
                            parent_entry.get("current_segment_id", added_entry.get("current_segment_id", ""))
                        ),
                        "next_segment_id": str(
                            parent_entry.get("next_segment_id", added_entry.get("next_segment_id", ""))
                        ),
                        "target_predicate": str(
                            parent_entry.get("target_predicate", added_entry.get("target_predicate", ""))
                        ),
                        "label": bool(parent_entry.get("label", added_entry.get("label", False))),
                        "bindings": _merge_bindings(parent_bindings, added_bindings),
                        "matched_atoms": matched_atoms,
                    }
                )

    return _dedupe_evidence_entries(intersected)


def _summarize_evidence(
    evidence_entries: Sequence[Dict[str, Any]],
    total_positive_examples: int,
) -> Dict[str, Any]:
    total_firings = len(evidence_entries)
    positive_firings = sum(1 for entry in evidence_entries if bool(entry.get("label", False)))
    negative_firings = total_firings - positive_firings
    positive_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    negative_example_ids = sorted(
        {
            str(entry.get("example_id", ""))
            for entry in evidence_entries
            if not bool(entry.get("label", False)) and str(entry.get("example_id", ""))
        }
    )
    confidence = float(positive_firings / max(1, total_firings))
    recall = float(len(positive_example_ids) / max(1, total_positive_examples))
    return {
        "positive_support": len(positive_example_ids),
        "negative_support": len(negative_example_ids),
        "total_support": len(set(positive_example_ids) | set(negative_example_ids)),
        "positive_firings": positive_firings,
        "negative_firings": negative_firings,
        "total_firings": total_firings,
        "confidence": confidence,
        "recall": recall,
        "positive_example_ids": positive_example_ids,
        "negative_example_ids": negative_example_ids,
    }


def _seed_rule_evidence(rule: Dict[str, Any]) -> List[Dict[str, Any]]:
    seeded: List[Dict[str, Any]] = []
    for entry in list(rule.get("evidence_set", [])):
        body_atom_template = str(entry.get("body_atom_template", rule.get("body_atom_template", ""))).strip()
        matched_atom = str(entry.get("matched_atom", "")).strip()
        seeded.append(
            {
                "video_id": str(entry.get("video_id", "")),
                "example_id": str(entry.get("example_id", "")),
                "current_segment_id": str(entry.get("current_segment_id", "")),
                "next_segment_id": str(entry.get("next_segment_id", "")),
                "target_predicate": str(entry.get("target_predicate", "")),
                "label": bool(entry.get("label", False)),
                "bindings": {str(k): str(v) for k, v in dict(entry.get("bindings", {})).items()},
                "matched_atoms": {body_atom_template: matched_atom} if body_atom_template and matched_atom else {},
            }
        )
    return _dedupe_evidence_entries(seeded)


def _serialize_rule(
    rule_index: int,
    round_index: int,
    head_predicate: str,
    head_atom_template: str,
    body_atom_templates: Sequence[str],
    parent_rule_id: str,
    added_body_atom_template: str,
    source_initial_rule_id: str,
    evidence_set: Sequence[Dict[str, Any]],
    evidence_summary: Dict[str, Any],
    parent_confidence: float,
    parent_positive_support: int,
    prune_reason: str,
    prune_status: str,
    kept_after_prune: bool,
) -> Dict[str, Any]:
    clause = _build_clause(head_atom_template, body_atom_templates)
    rule_id = f"extended_round_{round_index}_rule_{rule_index:06d}"
    return {
        "rule_index": rule_index,
        "rule_id": rule_id,
        "round_index": round_index,
        "head_predicate": head_predicate,
        "head_atom_template": head_atom_template,
        "body_atom_templates": list(body_atom_templates),
        "body_length": len(body_atom_templates),
        "body_atom_template": body_atom_templates[0] if len(body_atom_templates) == 1 else "",
        "clause": clause,
        "parent_rule_id": parent_rule_id,
        "added_body_atom_template": added_body_atom_template,
        "source_initial_rule_id": source_initial_rule_id,
        "positive_support": int(evidence_summary.get("positive_support", 0)),
        "negative_support": int(evidence_summary.get("negative_support", 0)),
        "total_support": int(evidence_summary.get("total_support", 0)),
        "positive_firings": int(evidence_summary.get("positive_firings", 0)),
        "negative_firings": int(evidence_summary.get("negative_firings", 0)),
        "total_firings": int(evidence_summary.get("total_firings", 0)),
        "confidence": float(evidence_summary.get("confidence", 0.0)),
        "recall": float(evidence_summary.get("recall", 0.0)),
        "positive_example_ids": list(evidence_summary.get("positive_example_ids", [])),
        "negative_example_ids": list(evidence_summary.get("negative_example_ids", [])),
        "parent_confidence": float(parent_confidence),
        "parent_positive_support": int(parent_positive_support),
        "confidence_improvement": float(evidence_summary.get("confidence", 0.0)) - float(parent_confidence),
        "support_improvement": int(evidence_summary.get("positive_support", 0)) - int(parent_positive_support),
        "evaluation_status": "implemented",
        "prune_reason": prune_reason,
        "prune_status": prune_status,
        "kept_after_prune": kept_after_prune,
        "evidence_set": list(evidence_set),
    }


def _write_round_outputs(
    out_root: Path,
    round_index: int,
    payload: Dict[str, Any],
) -> Tuple[Path, Path]:
    json_path = out_root / f"extended_rules_round_{round_index}.json"
    csv_path = out_root / f"extended_rules_round_{round_index}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_index",
                "rule_id",
                "round_index",
                "head_predicate",
                "head_atom_template",
                "body_atom_templates",
                "body_length",
                "body_atom_template",
                "clause",
                "parent_rule_id",
                "added_body_atom_template",
                "source_initial_rule_id",
                "positive_support",
                "negative_support",
                "total_support",
                "positive_firings",
                "negative_firings",
                "total_firings",
                "confidence",
                "recall",
                "parent_confidence",
                "confidence_improvement",
                "evaluation_status",
                "prune_reason",
                "prune_status",
                "kept_after_prune",
            ],
        )
        writer.writeheader()
        for rule in payload.get("rules", []):
            writer.writerow(
                {
                    "rule_index": rule.get("rule_index", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "round_index": rule.get("round_index", ""),
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_templates": json.dumps(rule.get("body_atom_templates", [])),
                    "body_length": rule.get("body_length", 0),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "clause": rule.get("clause", ""),
                    "parent_rule_id": rule.get("parent_rule_id", ""),
                    "added_body_atom_template": rule.get("added_body_atom_template", ""),
                    "source_initial_rule_id": rule.get("source_initial_rule_id", ""),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
                    "confidence": rule.get("confidence", 0.0),
                    "recall": rule.get("recall", 0.0),
                    "parent_confidence": rule.get("parent_confidence", 0.0),
                    "confidence_improvement": rule.get("confidence_improvement", 0.0),
                    "evaluation_status": rule.get("evaluation_status", ""),
                    "prune_reason": rule.get("prune_reason", ""),
                    "prune_status": rule.get("prune_status", ""),
                    "kept_after_prune": rule.get("kept_after_prune", True),
                }
            )

    return json_path, csv_path


def _build_all_kept_rules(
    initial_rules: Sequence[Dict[str, Any]],
    rounds: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_kept_rules: List[Dict[str, Any]] = []

    for rule_index, rule in enumerate(initial_rules):
        initial_rule = dict(rule)
        initial_rule["kept_stage"] = "initial"
        initial_rule["kept_round_index"] = 0
        initial_rule["kept_rule_index"] = rule_index
        all_kept_rules.append(initial_rule)

    for round_payload in rounds:
        round_index = int(round_payload.get("round_index", 0))
        for rule in list(round_payload.get("rules", [])):
            kept_rule = dict(rule)
            kept_rule["kept_stage"] = "extended"
            kept_rule["kept_round_index"] = round_index
            kept_rule["kept_rule_index"] = int(kept_rule.get("rule_index", -1))
            all_kept_rules.append(kept_rule)

    return all_kept_rules


def process_rules(
    merged_initial_rules: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    num_rounds = int(cfg.get("num_rounds", 3))
    evaluation_strategy = str(cfg.get("evaluation_strategy", "binding_aware_intersection"))
    prune_strategies = _resolve_prune_strategies(cfg)
    min_positive_support_to_extend = int(cfg.get("min_positive_support_to_extend", 1))
    same_confidence_smaller_evidence_enabled = bool(
        cfg.get("same_confidence_smaller_evidence_enabled", True)
    )

    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "extended_rules_manifest.json"

    if not force_recompute and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if int(cached.get("version", 0)) == _EXTENDED_RULES_VERSION and _cfg_key_subset(
            cached.get("config", {})
        ) == _cfg_key_subset(
            {
                "num_rounds": num_rounds,
                "evaluation_strategy": evaluation_strategy,
                "prune_strategies": prune_strategies,
                "min_positive_support_to_extend": min_positive_support_to_extend,
                "same_confidence_smaller_evidence_enabled": same_confidence_smaller_evidence_enabled,
            }
        ):
            print(f"  [cache] loading {manifest_path.name}")
            return cached

    initial_rules = list(merged_initial_rules.get("rules", []))
    total_positive_examples = int(merged_initial_rules.get("num_positive_examples", 0))
    initial_rule_atoms: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    seen_initial_atoms: set[str] = set()
    for rule in initial_rules:
        body_atom_template = _normalize_atom_template(str(rule.get("body_atom_template", "")))
        if not body_atom_template or body_atom_template in seen_initial_atoms:
            continue
        seen_initial_atoms.add(body_atom_template)
        initial_rule_atoms.append(
            (
                body_atom_template,
                str(rule.get("rule_id", "")),
                _seed_rule_evidence(rule),
            )
        )

    current_rules = []
    for rule in initial_rules:
        seeded_rule = dict(rule)
        seeded_rule["body_atom_templates"] = list(_get_rule_body_atom_templates(rule))
        seeded_rule["evidence_set"] = _seed_rule_evidence(rule)
        current_rules.append(seeded_rule)

    print(
        "  initial_rules: "
        f"total={len(initial_rules)} | "
        f"unique_body_atoms={len(initial_rule_atoms)}"
    )
    print(f"  prune_strategies: {prune_strategies}")

    round_summaries: List[Dict[str, Any]] = []
    rounds: List[Dict[str, Any]] = []

    for round_index in range(1, num_rounds + 1):
        next_rule_map: Dict[Tuple[str, str, Tuple[str, ...]], Dict[str, Any]] = {}
        num_candidates_generated = 0
        num_pruned = 0
        num_pruned_low_evidence = 0
        num_pruned_empty_evidence = 0
        num_pruned_no_positive = 0
        num_pruned_same_firings_as_parent = 0
        num_pruned_same_confidence_smaller_evidence = 0
        num_parent_rules_skipped = 0
        for parent_rule in current_rules:
            head_predicate = str(parent_rule.get("head_predicate", ""))
            head_atom_template = str(parent_rule.get("head_atom_template", f"{head_predicate}(S)."))
            parent_rule_id = str(parent_rule.get("rule_id", ""))
            parent_body_atoms = _get_rule_body_atom_templates(parent_rule)
            parent_evidence_set = list(parent_rule.get("evidence_set", []))
            parent_confidence = float(parent_rule.get("confidence", 0.0))
            parent_positive_support = int(parent_rule.get("positive_support", 0))
            parent_positive_firings = int(parent_rule.get("positive_firings", 0))
            parent_negative_firings = int(parent_rule.get("negative_firings", 0))
            parent_total_firings = int(parent_rule.get("total_firings", 0))

            for initial_body_atom_template, source_initial_rule_id, initial_evidence_set in initial_rule_atoms:
                if initial_body_atom_template in parent_body_atoms:
                    continue

                new_body_atoms = _canonical_body_atoms(parent_body_atoms + (initial_body_atom_template,))
                key = (head_predicate, head_atom_template, new_body_atoms)
                if key in next_rule_map:
                    continue

                num_candidates_generated += 1
                intersected_evidence = _intersect_evidence_sets(parent_evidence_set, initial_evidence_set)

                evidence_summary = _summarize_evidence(intersected_evidence, total_positive_examples)
                evidence_summary["parent_confidence"] = parent_confidence
                evidence_summary["parent_positive_firings"] = parent_positive_firings
                evidence_summary["parent_negative_firings"] = parent_negative_firings
                evidence_summary["parent_total_firings"] = parent_total_firings
                evidence_summary["is_last_round"] = bool(round_index == num_rounds)
                evidence_summary["min_positive_support_to_extend"] = min_positive_support_to_extend
                evidence_summary["same_confidence_smaller_evidence_enabled"] = (
                    same_confidence_smaller_evidence_enabled
                )
                prune_decision = apply_pruning_strategies(
                    rule={
                        "head_predicate": head_predicate,
                        "head_atom_template": head_atom_template,
                        "body_atom_templates": list(new_body_atoms),
                        "parent_rule_id": parent_rule_id,
                        "added_body_atom_template": initial_body_atom_template,
                    },
                    metrics=evidence_summary,
                    strategies=prune_strategies,
                )
                if not bool(prune_decision.get("kept", False)):
                    num_pruned += 1
                    prune_reason = str(prune_decision.get("prune_reason", ""))
                    if prune_reason == "low_evidence":
                        num_pruned_low_evidence += 1
                    elif prune_reason == "empty_evidence":
                        num_pruned_empty_evidence += 1
                    elif prune_reason == "no_positive_firings":
                        num_pruned_no_positive += 1
                    elif prune_reason == "same_firings_as_parent":
                        num_pruned_same_firings_as_parent += 1
                    elif prune_reason == "same_confidence_smaller_evidence":
                        num_pruned_same_confidence_smaller_evidence += 1
                    continue

                next_rule_map[key] = _serialize_rule(
                    rule_index=-1,
                    round_index=round_index,
                    head_predicate=head_predicate,
                    head_atom_template=head_atom_template,
                    body_atom_templates=new_body_atoms,
                    parent_rule_id=parent_rule_id,
                    added_body_atom_template=initial_body_atom_template,
                    source_initial_rule_id=source_initial_rule_id,
                    evidence_set=intersected_evidence,
                    evidence_summary=evidence_summary,
                    parent_confidence=parent_confidence,
                    parent_positive_support=parent_positive_support,
                    prune_reason=str(prune_decision.get("prune_reason", "")),
                    prune_status=str(prune_decision.get("prune_status", "kept")),
                    kept_after_prune=bool(prune_decision.get("kept", True)),
                )

        round_rules = sorted(
            next_rule_map.values(),
            key=lambda rule: (
                int(rule.get("body_length", 0)),
                str(rule.get("clause", "")),
            ),
        )
        for rule_index, rule in enumerate(round_rules):
            rule["rule_index"] = rule_index
            rule["rule_id"] = f"extended_round_{round_index}_rule_{rule_index:06d}"

        round_payload: Dict[str, Any] = {
            "version": _EXTENDED_RULES_VERSION,
            "round_index": round_index,
            "config": {
                "num_rounds": num_rounds,
                "evaluation_strategy": evaluation_strategy,
                "prune_strategies": prune_strategies,
                "min_positive_support_to_extend": min_positive_support_to_extend,
                "same_confidence_smaller_evidence_enabled": same_confidence_smaller_evidence_enabled,
            },
            "num_candidates_generated": num_candidates_generated,
            "num_rules": len(round_rules),
            "evaluation": {
                "status": "implemented",
                "strategy": evaluation_strategy,
            },
            "pruning": {
                "status": "implemented",
                "strategies": prune_strategies,
                "parent_rules_skipped_num": num_parent_rules_skipped,
                "pruned_num_rules": num_pruned,
                "pruned_low_evidence": num_pruned_low_evidence,
                "pruned_empty_evidence": num_pruned_empty_evidence,
                "pruned_no_positive_firings": num_pruned_no_positive,
                "pruned_same_firings_as_parent": num_pruned_same_firings_as_parent,
                "pruned_same_confidence_smaller_evidence": num_pruned_same_confidence_smaller_evidence,
                "kept_num_rules": len(round_rules),
            },
            "rules": round_rules,
        }
        json_path, csv_path = _write_round_outputs(out_root, round_index, round_payload)

        round_summaries.append(
            {
                "round_index": round_index,
                "num_candidates_generated": num_candidates_generated,
                "num_rules": len(round_rules),
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "evaluation_status": "implemented",
                "prune_status": "implemented",
                "parent_rules_skipped_num": num_parent_rules_skipped,
                "pruned_num_rules": num_pruned,
                "pruned_low_evidence": num_pruned_low_evidence,
                "pruned_empty_evidence": num_pruned_empty_evidence,
                "pruned_no_positive_firings": num_pruned_no_positive,
                "pruned_same_firings_as_parent": num_pruned_same_firings_as_parent,
                "pruned_same_confidence_smaller_evidence": num_pruned_same_confidence_smaller_evidence,
            }
        )
        rounds.append(round_payload)
        current_rules = round_rules

        print(
            f"  round={round_index}: candidates={num_candidates_generated} | "
            f"kept={len(round_rules)}"
        )
        print(
            "    parent_rules_skipped: "
            f"total={num_parent_rules_skipped}"
        )
        print(
            "    pruned_by_strategy: "
            f"total={num_pruned} | "
            f"low_evidence={num_pruned_low_evidence} | "
            f"empty={num_pruned_empty_evidence} | "
            f"no_positive={num_pruned_no_positive} | "
            f"same_firings_as_parent={num_pruned_same_firings_as_parent} | "
            "same_confidence_smaller_evidence="
            f"{num_pruned_same_confidence_smaller_evidence}"
        )

        if not current_rules:
            break

    manifest = {
        "version": _EXTENDED_RULES_VERSION,
        "config": {
            "num_rounds": num_rounds,
            "evaluation_strategy": evaluation_strategy,
            "prune_strategies": prune_strategies,
            "min_positive_support_to_extend": min_positive_support_to_extend,
            "same_confidence_smaller_evidence_enabled": same_confidence_smaller_evidence_enabled,
        },
        "input_num_examples": int(merged_initial_rules.get("num_examples", 0)),
        "input_num_positive_examples": total_positive_examples,
        "input_num_negative_examples": int(merged_initial_rules.get("num_negative_examples", 0)),
        "input_num_initial_rules": len(initial_rules),
        "input_num_unique_initial_body_atoms": len(initial_rule_atoms),
        "num_rounds_completed": len(rounds),
        "num_all_kept_rules": len(initial_rules) + sum(len(r.get("rules", [])) for r in rounds),
        "all_kept_rules": _build_all_kept_rules(initial_rules, rounds),
        "round_summaries": round_summaries,
        "rounds": rounds,
    }

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Extended rules manifest written to {manifest_path}")
    return manifest


def run(
    merged_initial_rules: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_rules(
        merged_initial_rules=merged_initial_rules,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
