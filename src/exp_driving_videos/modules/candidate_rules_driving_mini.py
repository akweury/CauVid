"""
Generate temporal initial rules from step-13 rule-learning examples.

Current scope:
  - Only unary-body temporal clauses are generated.
  - Rule head is the target predicate, currently brake_next.
  - Rules are intended as short seed clauses for later extension steps.

Output layout:
    pipeline_output/14_driving_mini_initial_rules/
        initial_rules_manifest.json
        <video_id>/
            initial_rules.json
            initial_rules.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_INITIAL_RULES_VERSION = 8
_SINGLETON_PROVENANCE_ONLY_PREDICATES = {
    "object_is_candidate",
    "object_source_type",
    "object_candidate_score_state",
    "object_prior_relevance_state",
    "object_matched_prior",
}
_VARIABLE_ARGS = {"S", "O", "C", "T", "F"}


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "14_driving_mini_initial_rules"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "target_predicate",
        "negative_target_predicate",
        "use_only_positive_examples",
        "min_positive_support",
        "include_example_ids",
        "ignored_body_predicates",
        "generate_mixed_accepted_candidate_rules",
    ]
    return {k: cfg.get(k) for k in keys}


def _print_video_summary(result: Dict[str, Any]) -> None:
    print(
        f"  {result.get('video_id', 'unknown')}: "
        f"initial_rules={int(result.get('num_initial_rules', result.get('num_candidate_rules', 0)))} | "
        f"target={result.get('target_predicate', 'unknown')}"
    )


def _parse_atom(atom: str) -> Optional[Tuple[str, List[str]]]:
    text = str(atom).strip()
    m = re.match(r"^([a-z0-9_]+)\((.*)\)\.$", text)
    if not m:
        return None
    predicate = m.group(1)
    args_text = m.group(2).strip()
    if not args_text:
        return predicate, []
    args = [part.strip() for part in args_text.split(",")]
    return predicate, args


def _abstract_arg(arg: str) -> str:
    text = str(arg)
    if text.startswith("seg_"):
        return "S"
    if text.startswith("obj_candidate_"):
        return "C"
    if text.startswith("obj_"):
        return "O"
    if text.startswith("track_"):
        return "T"
    if text.startswith("frame_"):
        return "F"
    return text


def _abstract_atom(atom: str) -> Optional[str]:
    parsed = _parse_atom(atom)
    if parsed is None:
        return None
    predicate, args = parsed
    abstract_args = [_abstract_arg(arg) for arg in args]
    return f"{predicate}({','.join(abstract_args)})."


def _build_clause(head_predicate: str, body_atom_template: str) -> str:
    head = f"{head_predicate}(S)."
    head_no_dot = head[:-1]
    return f"{head_no_dot} :- {body_atom_template}"


def _extract_bindings(
    body_atom_template: str,
    concrete_atom: str,
) -> Optional[Dict[str, str]]:
    template_parsed = _parse_atom(body_atom_template)
    concrete_parsed = _parse_atom(concrete_atom)
    if template_parsed is None or concrete_parsed is None:
        return None

    template_predicate, template_args = template_parsed
    concrete_predicate, concrete_args = concrete_parsed
    if template_predicate != concrete_predicate or len(template_args) != len(concrete_args):
        return None

    bindings: Dict[str, str] = {}
    for template_arg, concrete_arg in zip(template_args, concrete_args):
        if template_arg in _VARIABLE_ARGS:
            existing = bindings.get(template_arg)
            if existing is not None and existing != concrete_arg:
                return None
            bindings[template_arg] = concrete_arg
            continue
        if template_arg != concrete_arg:
            return None
    return bindings


def _make_evidence_entry(
    video_id: str,
    example: Dict[str, Any],
    body_atom_template: str,
    concrete_atom: str,
    body_atom_source: str,
    matched_prior_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    bindings = _extract_bindings(body_atom_template, concrete_atom)
    if bindings is None:
        return None

    return {
        "video_id": video_id,
        "example_id": str(example.get("example_id", "")),
        "current_segment_id": str(example.get("current_segment_id", "")),
        "next_segment_id": str(example.get("next_segment_id", "")),
        "target_predicate": str(example.get("target_predicate", "")),
        "label": bool(example.get("label", False)),
        "body_atom_template": body_atom_template,
        "matched_atom": str(concrete_atom),
        "body_atom_source": str(body_atom_source),
        "matched_prior_ids": sorted({str(value) for value in list(matched_prior_ids or []) if str(value)}),
        "bindings": bindings,
        "matched_atoms": {body_atom_template: str(concrete_atom)},
        "matched_atom_sources": {body_atom_template: str(body_atom_source)},
        "matched_atom_prior_ids": {
            body_atom_template: sorted({str(value) for value in list(matched_prior_ids or []) if str(value)})
        },
    }


def _dedupe_evidence_entries(evidence_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, Tuple[Tuple[str, str], ...]]] = set()
    for entry in evidence_entries:
        key = (
            str(entry.get("example_id", "")),
            str(entry.get("matched_atom", "")),
            tuple(sorted((str(k), str(v)) for k, v in dict(entry.get("bindings", {})).items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _summarize_evidence(
    evidence_entries: List[Dict[str, Any]],
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
    total_example_ids = sorted(set(positive_example_ids) | set(negative_example_ids))
    confidence = float(positive_firings / max(1, total_firings))

    return {
        "positive_support": len(positive_example_ids),
        "negative_support": len(negative_example_ids),
        "total_support": len(total_example_ids),
        "positive_firings": positive_firings,
        "negative_firings": negative_firings,
        "total_firings": total_firings,
        "confidence": confidence,
        "positive_example_ids": positive_example_ids,
        "negative_example_ids": negative_example_ids,
    }


def _summarize_rule_candidate_provenance(
    *,
    evidence_entries: List[Dict[str, Any]],
    body_length: int,
) -> Dict[str, Any]:
    candidate_body_templates: set[str] = set()
    accepted_body_templates: set[str] = set()
    candidate_evidence_firings = 0
    accepted_evidence_firings = 0
    matched_prior_ids_involved = sorted(
        {
            str(prior_id)
            for entry in evidence_entries
            for prior_ids in dict(entry.get("matched_atom_prior_ids", {})).values()
            for prior_id in list(prior_ids)
            if str(prior_id)
        }
    )
    for entry in evidence_entries:
        matched_atom_sources = dict(entry.get("matched_atom_sources", {}))
        if not matched_atom_sources:
            body_atom_template = str(entry.get("body_atom_template", "")).strip()
            source = str(entry.get("body_atom_source", "")).strip()
            if body_atom_template and source:
                matched_atom_sources = {body_atom_template: source}
        for body_atom_template, source in matched_atom_sources.items():
            if str(source) == "candidate":
                candidate_body_templates.add(str(body_atom_template))
                candidate_evidence_firings += 1
            elif str(source) == "accepted":
                accepted_body_templates.add(str(body_atom_template))
                accepted_evidence_firings += 1

    uses_candidate_atoms = bool(candidate_body_templates)
    uses_only_candidate_atoms = uses_candidate_atoms and not accepted_body_templates
    mixes_accepted_and_candidate_atoms = bool(candidate_body_templates) and bool(accepted_body_templates)
    num_candidate_body_atoms = len(candidate_body_templates)
    if mixes_accepted_and_candidate_atoms:
        body_source_mix = "mixed_accepted_and_candidate"
    elif uses_only_candidate_atoms:
        body_source_mix = "candidate_only"
    elif accepted_body_templates:
        body_source_mix = "accepted_only"
    else:
        body_source_mix = "non_object_or_segment_only"

    return {
        "uses_candidate_atoms": uses_candidate_atoms,
        "num_candidate_body_atoms": num_candidate_body_atoms,
        "candidate_body_atom_ratio": float(num_candidate_body_atoms / max(1, int(body_length))),
        "matched_prior_ids_involved": matched_prior_ids_involved,
        "mixes_accepted_and_candidate_atoms": mixes_accepted_and_candidate_atoms,
        "uses_only_candidate_atoms": uses_only_candidate_atoms,
        "body_source_mix": body_source_mix,
        "candidate_evidence_firings": candidate_evidence_firings,
        "accepted_evidence_firings": accepted_evidence_firings,
    }


def _rule_category_from_provenance(provenance_summary: Dict[str, Any]) -> str:
    if bool(provenance_summary.get("mixes_accepted_and_candidate_atoms", False)):
        return "mixed_accepted_candidate"
    if bool(provenance_summary.get("uses_only_candidate_atoms", False)):
        return "candidate_only"
    if bool(provenance_summary.get("uses_candidate_atoms", False)):
        return "candidate_only"
    return "accepted_only"


def _empty_category_counts() -> Dict[str, int]:
    return {
        "accepted_only_rules": 0,
        "candidate_only_rules": 0,
        "mixed_accepted_candidate_rules": 0,
        "candidate_involving_rules": 0,
        "all_rules": 0,
    }


def _count_rule_categories(rules: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = _empty_category_counts()
    for rule in rules:
        category = str(rule.get("candidate_rule_category", ""))
        if not category:
            category = _rule_category_from_provenance(rule)
        counts["all_rules"] += 1
        if category == "mixed_accepted_candidate":
            counts["mixed_accepted_candidate_rules"] += 1
            counts["candidate_involving_rules"] += 1
        elif category == "candidate_only":
            counts["candidate_only_rules"] += 1
            counts["candidate_involving_rules"] += 1
        else:
            counts["accepted_only_rules"] += 1
    return counts


def _predicate_of_atom_template(atom_text: str) -> str:
    text = str(atom_text).strip().rstrip(".").strip()
    if "(" not in text:
        return text
    return text.split("(", 1)[0].strip()


def _is_provenance_only_atom_template(atom_text: str) -> bool:
    return _predicate_of_atom_template(atom_text) in _SINGLETON_PROVENANCE_ONLY_PREDICATES


def _bindings_compatible(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    for key, value in left.items():
        if key in right and str(right[key]) != str(value):
            return False
    return True


def _merge_bindings(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, str]:
    merged = {str(key): str(value) for key, value in left.items()}
    for key, value in right.items():
        merged[str(key)] = str(value)
    return merged


def _combine_evidence_entries(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not entries:
        return None
    combined_bindings: Dict[str, str] = {}
    matched_atoms: Dict[str, str] = {}
    matched_atom_sources: Dict[str, str] = {}
    matched_atom_prior_ids: Dict[str, List[str]] = {}
    for entry in entries:
        entry_bindings = {str(k): str(v) for k, v in dict(entry.get("bindings", {})).items()}
        if not _bindings_compatible(combined_bindings, entry_bindings):
            return None
        combined_bindings = _merge_bindings(combined_bindings, entry_bindings)
        matched_atoms.update({str(k): str(v) for k, v in dict(entry.get("matched_atoms", {})).items()})
        matched_atom_sources.update(
            {str(k): str(v) for k, v in dict(entry.get("matched_atom_sources", {})).items()}
        )
        for key, prior_ids in dict(entry.get("matched_atom_prior_ids", {})).items():
            out_prior_ids = matched_atom_prior_ids.setdefault(str(key), [])
            for prior_id in list(prior_ids):
                prior_id_text = str(prior_id)
                if prior_id_text and prior_id_text not in out_prior_ids:
                    out_prior_ids.append(prior_id_text)

    first = entries[0]
    return {
        "video_id": str(first.get("video_id", "")),
        "example_id": str(first.get("example_id", "")),
        "current_segment_id": str(first.get("current_segment_id", "")),
        "next_segment_id": str(first.get("next_segment_id", "")),
        "target_predicate": str(first.get("target_predicate", "")),
        "label": bool(first.get("label", False)),
        "body_atom_templates": sorted(matched_atoms),
        "body_atom_template": sorted(matched_atoms)[0] if len(matched_atoms) == 1 else "",
        "matched_atom": next(iter(matched_atoms.values()), "") if len(matched_atoms) == 1 else "",
        "body_atom_source": next(iter(matched_atom_sources.values()), "") if len(matched_atom_sources) == 1 else "",
        "matched_prior_ids": sorted(
            {
                str(prior_id)
                for prior_ids in matched_atom_prior_ids.values()
                for prior_id in list(prior_ids)
                if str(prior_id)
            }
        ),
        "bindings": combined_bindings,
        "matched_atoms": matched_atoms,
        "matched_atom_sources": matched_atom_sources,
        "matched_atom_prior_ids": {
            key: sorted(values)
            for key, values in matched_atom_prior_ids.items()
        },
    }


def _canonical_body_templates(body_templates: List[str]) -> Tuple[str, ...]:
    return tuple(sorted({str(template).strip() for template in body_templates if str(template).strip()}))


def _build_clause_from_templates(head_predicate: str, body_atom_templates: Tuple[str, ...]) -> str:
    head_no_dot = f"{head_predicate}(S)"
    body = ", ".join(str(atom).rstrip(".") for atom in body_atom_templates)
    return f"{head_no_dot} :- {body}."


def process_video(
    temporal_rule_examples_video_result: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    target_predicate = str(cfg.get("target_predicate", "brake_next"))
    negative_target_predicate = str(cfg.get("negative_target_predicate", f"not_{target_predicate}"))
    use_only_positive_examples = bool(cfg.get("use_only_positive_examples", True))
    min_positive_support = int(cfg.get("min_positive_support", 1))
    include_example_ids = bool(cfg.get("include_example_ids", True))
    generate_mixed_accepted_candidate_rules = bool(cfg.get("generate_mixed_accepted_candidate_rules", True))
    ignored_body_predicates = {
        str(name).strip()
        for name in cfg.get(
            "ignored_body_predicates",
            [
                "segment",
                "segment_start_frame",
                "segment_end_frame",
                "object_in_segment",
                "object_track",
            ],
        )
        if str(name).strip()
    }

    video_id = str(temporal_rule_examples_video_result["video_id"])
    out_dir = (output_root or get_output_root()) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "initial_rules.json"
    csv_file = out_dir / "initial_rules.csv"

    if not force_recompute and out_file.exists():
        with out_file.open("r", encoding="utf-8") as fh:
            cached = json.load(fh)
        cache_cfg = cached.get("config", {}) if isinstance(cached, dict) else {}
        if (
            int(cached.get("version", 0)) == _INITIAL_RULES_VERSION
            and _cfg_key_subset(cache_cfg) == _cfg_key_subset(
                {
                    "target_predicate": target_predicate,
                    "negative_target_predicate": negative_target_predicate,
                    "use_only_positive_examples": use_only_positive_examples,
                    "min_positive_support": min_positive_support,
                    "include_example_ids": include_example_ids,
                    "ignored_body_predicates": sorted(ignored_body_predicates),
                }
            )
        ):
            print(f"  [cache] {video_id} - loading {out_file.name}")
            _print_video_summary(cached)
            return cached

    examples = list(temporal_rule_examples_video_result.get("examples", []))
    stats: Dict[Tuple[str, ...], Dict[str, Any]] = defaultdict(lambda: {"evidence_set": []})
    eligible_body_template_sets: set[Tuple[str, ...]] = set()
    available_body_templates: set[str] = set()
    available_candidate_body_templates: set[str] = set()
    available_candidate_semantic_body_templates: set[str] = set()
    available_candidate_provenance_only_body_templates: set[str] = set()
    available_accepted_body_templates: set[str] = set()
    total_positive_examples = sum(1 for example in examples if bool(example.get("label", False)))
    total_negative_examples = sum(1 for example in examples if not bool(example.get("label", False)))

    for example in examples:
        example_target_predicate = str(example.get("target_predicate", ""))
        if example_target_predicate not in {target_predicate, negative_target_predicate}:
            continue

        is_positive = bool(example.get("label", False))
        example_id = str(example.get("example_id", ""))
        body_atoms = list(example.get("body_atoms", []))
        accepted_body_atoms = {str(atom) for atom in list(example.get("accepted_body_atoms", []))}
        candidate_body_atoms = {str(atom) for atom in list(example.get("candidate_body_atoms", []))}
        matched_prior_ids_by_candidate_object: Dict[str, List[str]] = defaultdict(list)
        for candidate_atom in candidate_body_atoms:
            parsed_candidate_atom = _parse_atom(candidate_atom)
            if parsed_candidate_atom is None:
                continue
            candidate_predicate, candidate_args = parsed_candidate_atom
            if candidate_predicate != "object_matched_prior" or len(candidate_args) < 2:
                continue
            object_id = str(candidate_args[0])
            prior_id = str(candidate_args[1])
            if prior_id and prior_id not in matched_prior_ids_by_candidate_object[object_id]:
                matched_prior_ids_by_candidate_object[object_id].append(prior_id)
        unary_entries_by_template: Dict[str, Dict[str, Any]] = {}
        accepted_entries: List[Dict[str, Any]] = []
        candidate_entries: List[Dict[str, Any]] = []
        for body_atom in body_atoms:
            parsed_body_atom = _parse_atom(str(body_atom))
            if parsed_body_atom is None:
                continue
            body_predicate, _ = parsed_body_atom
            if body_predicate in ignored_body_predicates:
                continue
            body_template = _abstract_atom(str(body_atom))
            if not body_template or body_template in unary_entries_by_template:
                continue
            body_atom_source = (
                "candidate"
                if str(body_atom) in candidate_body_atoms
                else ("accepted" if str(body_atom) in accepted_body_atoms else "segment_or_other")
            )
            matched_prior_ids: List[str] = []
            bindings = _extract_bindings(body_template, str(body_atom))
            if body_atom_source == "candidate" and isinstance(bindings, dict):
                matched_prior_ids = list(matched_prior_ids_by_candidate_object.get(str(bindings.get("O", "")), []))
            evidence_entry = _make_evidence_entry(
                video_id=video_id,
                example=example,
                body_atom_template=body_template,
                concrete_atom=str(body_atom),
                body_atom_source=body_atom_source,
                matched_prior_ids=matched_prior_ids,
            )
            if evidence_entry is not None:
                unary_entries_by_template[body_template] = evidence_entry
                unary_key = (body_template,)
                stats[unary_key]["evidence_set"].append(evidence_entry)
                available_body_templates.add(body_template)
                if body_atom_source == "candidate":
                    candidate_entries.append(evidence_entry)
                    available_candidate_body_templates.add(body_template)
                    if _is_provenance_only_atom_template(body_template):
                        available_candidate_provenance_only_body_templates.add(body_template)
                    else:
                        available_candidate_semantic_body_templates.add(body_template)
                elif body_atom_source == "accepted":
                    accepted_entries.append(evidence_entry)
                    available_accepted_body_templates.add(body_template)

        if generate_mixed_accepted_candidate_rules:
            for left_index, left_entry in enumerate(candidate_entries):
                left_template = str(left_entry.get("body_atom_template", "")).strip()
                if not left_template:
                    continue
                for right_entry in candidate_entries[left_index + 1 :]:
                    right_template = str(right_entry.get("body_atom_template", "")).strip()
                    if (
                        not right_template
                        or right_template == left_template
                        or (
                            _is_provenance_only_atom_template(left_template)
                            and _is_provenance_only_atom_template(right_template)
                        )
                    ):
                        continue
                    body_templates = _canonical_body_templates([left_template, right_template])
                    if len(body_templates) != 2:
                        continue
                    combined_entry = _combine_evidence_entries([left_entry, right_entry])
                    if combined_entry is not None:
                        stats[body_templates]["evidence_set"].append(combined_entry)

            for accepted_entry in accepted_entries:
                accepted_template = str(accepted_entry.get("body_atom_template", "")).strip()
                if not accepted_template:
                    continue
                for candidate_entry in candidate_entries:
                    candidate_template = str(candidate_entry.get("body_atom_template", "")).strip()
                    if (
                        not candidate_template
                        or candidate_template == accepted_template
                    ):
                        continue
                    body_templates = _canonical_body_templates([accepted_template, candidate_template])
                    if len(body_templates) != 2:
                        continue
                    combined_entry = _combine_evidence_entries([accepted_entry, candidate_entry])
                    if combined_entry is not None:
                        stats[body_templates]["evidence_set"].append(combined_entry)

        if is_positive or not use_only_positive_examples:
            for body_template, entry in unary_entries_by_template.items():
                if _is_provenance_only_atom_template(body_template):
                    continue
                eligible_body_template_sets.add((body_template,))
            if generate_mixed_accepted_candidate_rules:
                for left_index, left_entry in enumerate(candidate_entries):
                    left_template = str(left_entry.get("body_atom_template", "")).strip()
                    if not left_template:
                        continue
                    for right_entry in candidate_entries[left_index + 1 :]:
                        right_template = str(right_entry.get("body_atom_template", "")).strip()
                        if (
                            not right_template
                            or right_template == left_template
                            or (
                                _is_provenance_only_atom_template(left_template)
                                and _is_provenance_only_atom_template(right_template)
                            )
                        ):
                            continue
                        body_templates = _canonical_body_templates([left_template, right_template])
                        if len(body_templates) == 2:
                            eligible_body_template_sets.add(body_templates)

                for accepted_entry in accepted_entries:
                    accepted_template = str(accepted_entry.get("body_atom_template", "")).strip()
                    if not accepted_template:
                        continue
                    for candidate_entry in candidate_entries:
                        candidate_template = str(candidate_entry.get("body_atom_template", "")).strip()
                        if (
                            not candidate_template
                            or candidate_template == accepted_template
                        ):
                            continue
                        body_templates = _canonical_body_templates([accepted_template, candidate_template])
                        if len(body_templates) == 2:
                            eligible_body_template_sets.add(body_templates)

    initial_rules: List[Dict[str, Any]] = []
    for idx, body_templates in enumerate(sorted(stats.keys())):
        if body_templates not in eligible_body_template_sets:
            continue

        evidence_set = _dedupe_evidence_entries(list(stats[body_templates].get("evidence_set", [])))
        evidence_summary = _summarize_evidence(evidence_set)
        provenance_summary = _summarize_rule_candidate_provenance(
            evidence_entries=evidence_set,
            body_length=len(body_templates),
        )
        positive_support = int(evidence_summary["positive_support"])
        negative_support = int(evidence_summary["negative_support"])
        total_support = int(evidence_summary["total_support"])
        if positive_support < min_positive_support:
            continue

        confidence = float(evidence_summary["confidence"])
        clause = _build_clause_from_templates(target_predicate, body_templates)
        candidate_rule_category = _rule_category_from_provenance(provenance_summary)
        initial_rules.append(
            {
                "rule_index": len(initial_rules),
                "rule_id": f"{video_id}_rule_{idx:04d}",
                "head_predicate": target_predicate,
                "head_atom_template": f"{target_predicate}(S).",
                "body_atom_template": body_templates[0] if len(body_templates) == 1 else "",
                "body_atom_templates": list(body_templates),
                "body_length": len(body_templates),
                "clause": clause,
                "positive_support": positive_support,
                "negative_support": negative_support,
                "total_support": total_support,
                "positive_firings": int(evidence_summary["positive_firings"]),
                "negative_firings": int(evidence_summary["negative_firings"]),
                "total_firings": int(evidence_summary["total_firings"]),
                "confidence": confidence,
                "positive_example_ids": (
                    evidence_summary["positive_example_ids"] if include_example_ids else []
                ),
                "negative_example_ids": (
                    evidence_summary["negative_example_ids"] if include_example_ids else []
                ),
                **provenance_summary,
                "candidate_rule_category": candidate_rule_category,
                "evidence_set": evidence_set,
            }
        )

    generated_category_counts = _count_rule_categories(initial_rules)
    result: Dict[str, Any] = {
        "version": _INITIAL_RULES_VERSION,
        "video_id": video_id,
        "target_predicate": target_predicate,
        "num_initial_rules": len(initial_rules),
        "num_examples": len(examples),
        "num_positive_examples": total_positive_examples,
        "num_negative_examples": total_negative_examples,
        "num_rules_using_candidate_atoms": generated_category_counts["candidate_involving_rules"],
        "num_candidate_only_rules": generated_category_counts["candidate_only_rules"],
        "num_mixed_source_rules": generated_category_counts["mixed_accepted_candidate_rules"],
        "candidate_rule_stage_stats": {
            "stage": "step14_initial_generation",
            "atom_availability": {
                "num_body_atom_templates": len(available_body_templates),
                "num_accepted_body_atom_templates": len(available_accepted_body_templates),
                "num_candidate_body_atom_templates": len(available_candidate_body_templates),
                "num_candidate_semantic_body_atom_templates": len(available_candidate_semantic_body_templates),
                "num_candidate_provenance_only_body_atom_templates": len(
                    available_candidate_provenance_only_body_templates
                ),
                "candidate_semantic_body_atom_templates": sorted(available_candidate_semantic_body_templates),
                "candidate_provenance_only_body_atom_templates": sorted(
                    available_candidate_provenance_only_body_templates
                ),
            },
            "generated_rule_counts": generated_category_counts,
        },
        "candidate_rule_flow_summary": {
            "atom_availability": {
                "num_candidate_body_atom_templates": len(available_candidate_body_templates),
                "num_candidate_semantic_body_atom_templates": len(available_candidate_semantic_body_templates),
                "num_candidate_provenance_only_body_atom_templates": len(
                    available_candidate_provenance_only_body_templates
                ),
            },
            "initial_rule_generation": generated_category_counts,
            "pruning": _empty_category_counts(),
            "extension": _empty_category_counts(),
            "final_selection": _empty_category_counts(),
            "evaluation": _empty_category_counts(),
        },
        "config": {
            "target_predicate": target_predicate,
            "negative_target_predicate": negative_target_predicate,
            "use_only_positive_examples": use_only_positive_examples,
            "min_positive_support": min_positive_support,
            "include_example_ids": include_example_ids,
            "ignored_body_predicates": sorted(ignored_body_predicates),
            "generate_mixed_accepted_candidate_rules": generate_mixed_accepted_candidate_rules,
        },
        "initial_rules": initial_rules,
    }

    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rule_index",
                "rule_id",
                "head_predicate",
                "head_atom_template",
                "body_atom_template",
                "body_atom_templates",
                "body_length",
                "clause",
                "positive_support",
                "negative_support",
                "total_support",
                "positive_firings",
                "negative_firings",
                "total_firings",
                "confidence",
                "uses_candidate_atoms",
                "num_candidate_body_atoms",
                "candidate_body_atom_ratio",
                "matched_prior_ids_involved",
                "mixes_accepted_and_candidate_atoms",
                "uses_only_candidate_atoms",
                "body_source_mix",
                "candidate_rule_category",
                "positive_example_ids",
                "negative_example_ids",
            ],
        )
        writer.writeheader()
        for rule in initial_rules:
            writer.writerow(
                {
                    "rule_index": rule.get("rule_index", ""),
                    "rule_id": rule.get("rule_id", ""),
                    "head_predicate": rule.get("head_predicate", ""),
                    "head_atom_template": rule.get("head_atom_template", ""),
                    "body_atom_template": rule.get("body_atom_template", ""),
                    "body_atom_templates": json.dumps(rule.get("body_atom_templates", [])),
                    "body_length": rule.get("body_length", 1),
                    "clause": rule.get("clause", ""),
                    "positive_support": rule.get("positive_support", 0),
                    "negative_support": rule.get("negative_support", 0),
                    "total_support": rule.get("total_support", 0),
                    "positive_firings": rule.get("positive_firings", 0),
                    "negative_firings": rule.get("negative_firings", 0),
                    "total_firings": rule.get("total_firings", 0),
                    "confidence": rule.get("confidence", 0.0),
                    "uses_candidate_atoms": rule.get("uses_candidate_atoms", False),
                    "num_candidate_body_atoms": rule.get("num_candidate_body_atoms", 0),
                    "candidate_body_atom_ratio": rule.get("candidate_body_atom_ratio", 0.0),
                    "matched_prior_ids_involved": json.dumps(rule.get("matched_prior_ids_involved", [])),
                    "mixes_accepted_and_candidate_atoms": rule.get("mixes_accepted_and_candidate_atoms", False),
                    "uses_only_candidate_atoms": rule.get("uses_only_candidate_atoms", False),
                    "body_source_mix": rule.get("body_source_mix", ""),
                    "candidate_rule_category": rule.get("candidate_rule_category", ""),
                    "positive_example_ids": json.dumps(rule.get("positive_example_ids", [])),
                    "negative_example_ids": json.dumps(rule.get("negative_example_ids", [])),
                }
            )

    _print_video_summary(result)
    return result


def run(
    temporal_rule_results: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    out_root = output_root or get_output_root()
    results: List[Dict[str, Any]] = []

    for temporal_rule_result in temporal_rule_results:
        result = process_video(
            temporal_rule_examples_video_result=temporal_rule_result,
            cfg=cfg,
            output_root=out_root,
            force_recompute=force_recompute,
        )
        results.append(result)

    aggregate_generated_counts = _empty_category_counts()
    aggregate_atom_availability = {
        "num_body_atom_templates": 0,
        "num_accepted_body_atom_templates": 0,
        "num_candidate_body_atom_templates": 0,
        "num_candidate_semantic_body_atom_templates": 0,
        "num_candidate_provenance_only_body_atom_templates": 0,
    }
    candidate_semantic_body_atom_templates: set[str] = set()
    candidate_provenance_only_body_atom_templates: set[str] = set()
    for result in results:
        stage_stats = dict(result.get("candidate_rule_stage_stats", {}))
        generated_counts = dict(stage_stats.get("generated_rule_counts", {}))
        for key in aggregate_generated_counts:
            aggregate_generated_counts[key] += int(generated_counts.get(key, 0))
        atom_availability = dict(stage_stats.get("atom_availability", {}))
        for key in aggregate_atom_availability:
            aggregate_atom_availability[key] += int(atom_availability.get(key, 0))
        candidate_semantic_body_atom_templates.update(
            str(value)
            for value in list(atom_availability.get("candidate_semantic_body_atom_templates", []))
            if str(value)
        )
        candidate_provenance_only_body_atom_templates.update(
            str(value)
            for value in list(atom_availability.get("candidate_provenance_only_body_atom_templates", []))
            if str(value)
        )
    aggregate_atom_availability["candidate_semantic_body_atom_templates"] = sorted(
        candidate_semantic_body_atom_templates
    )
    aggregate_atom_availability["candidate_provenance_only_body_atom_templates"] = sorted(
        candidate_provenance_only_body_atom_templates
    )

    manifest = {
        "version": _INITIAL_RULES_VERSION,
        "num_videos": len(results),
        "candidate_rule_stage_stats": {
            "stage": "step14_initial_generation",
            "atom_availability": aggregate_atom_availability,
            "generated_rule_counts": aggregate_generated_counts,
        },
        "videos": [
            {
                "video_id": r["video_id"],
                "target_predicate": r.get("target_predicate", ""),
                "num_initial_rules": r.get("num_initial_rules", r.get("num_candidate_rules", 0)),
                "num_examples": r.get("num_examples", 0),
                "num_positive_examples": r.get("num_positive_examples", 0),
                "num_negative_examples": r.get("num_negative_examples", 0),
                "num_rules_using_candidate_atoms": r.get("num_rules_using_candidate_atoms", 0),
                "num_candidate_only_rules": r.get("num_candidate_only_rules", 0),
                "num_mixed_source_rules": r.get("num_mixed_source_rules", 0),
            }
            for r in results
        ],
    }
    manifest_path = out_root / "initial_rules_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Initial rules manifest written to {manifest_path}")
    return results
