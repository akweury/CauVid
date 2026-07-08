"""
Causal masking-guided reselection of an existing final rule set.

This step does not mine rules, create facts, modify detections, or modify
tracking. It only re-ranks and filters Step 17 rules using Step 18/18M
validation diagnostics.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import config


_RESELECTION_VERSION = 6
_ROW_FIELDS = [
    "rule_id",
    "clause",
    "original_clause",
    "original_rank",
    "original_score",
    "found_in_step18m",
    "missing_from_step18m",
    "found_in_step17",
    "matched_by",
    "causal_effect_status",
    "alignment_warning",
    "selection_source",
    "backfill_reason",
    "helpful_count",
    "harmful_count",
    "necessary_true_positive_count",
    "causal_false_positive_count",
    "redundant_count",
    "prediction_flip_count",
    "net_helpful_minus_harmful",
    "dominant_influence_type",
    "assigned_reselection_category",
    "reselection_decision",
    "reason",
    "trigger_count",
    "non_decisive_contribution_count",
    "weak_grounding_penalty_applied",
    "has_object_level_support",
    "uses_only_ego_motion_atoms",
    "uses_broad_weak_atoms",
    "reasoning_feedback_request_count",
    "backup_explanation_candidate",
    "warning_message",
    "blacklist_status",
    "removal_reason",
    "refill_reason",
    "refill_rank",
    "replaced_removed_rule_id",
    "reselection_score",
]
_REFINED_RULE_FIELDS = [
    "rule_id",
    "clause",
    "original_rank",
    "original_score",
    "selection_source",
    "reselection_decision",
    "assigned_reselection_category",
    "helpful_count",
    "harmful_count",
    "necessary_true_positive_count",
    "causal_false_positive_count",
    "prediction_flip_count",
    "found_in_step18m",
    "blacklist_status",
    "refill_reason",
    "replaced_removed_rule_id",
    "refined_rank",
]
_REMOVED_RULE_FIELDS = [
    "rule_id",
    "clause",
    "original_rank",
    "removal_reason",
    "helpful_count",
    "harmful_count",
    "necessary_true_positive_count",
    "causal_false_positive_count",
    "prediction_flip_count",
]
_REFILLED_RULE_FIELDS = [
    "rule_id",
    "clause",
    "original_rank",
    "original_score",
    "refill_reason",
    "replaced_removed_rule_id",
    "selection_source",
]
_REFINEMENT_TARGET_FIELDS = _ROW_FIELDS + ["suggested_action"]
_PER_ROUND_EVALUATION_FIELDS = [
    "round_index",
    "rule_set_name",
    "num_rules",
    "num_removed_rules",
    "num_refilled_rules",
    "num_blacklisted_rules",
    "top_k_reached",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "selected_as_best",
    "early_stop_reason",
]
_DESCENDANT_CANDIDATE_FIELDS = [
    "round_index",
    "parent_rule_id",
    "parent_clause",
    "candidate_rule_id",
    "candidate_clause",
    "candidate_rank",
    "score",
    "precision_gain",
    "fp_reduction",
    "positive_coverage_retention",
    "causal_grounding_bonus",
    "rank_bonus",
    "rule_length_penalty",
    "parent_reason",
]
_DESCENDANT_TRIAL_FIELDS = [
    "round_index",
    "parent_rule_id",
    "descendant_rule_id",
    "parent_clause",
    "descendant_clause",
    "trial_status",
    "validation_f1",
    "validation_precision",
    "improved_over_previous_best",
    "failure_reason",
]
_DESCENDANT_ROUND_SUMMARY_FIELDS = [
    "round_index",
    "num_candidate_parents",
    "num_candidate_descendants",
    "num_trial_replacements",
    "num_failed_trials",
    "validation_f1",
    "validation_precision",
    "selected_as_best",
]


def get_output_root() -> Path:
    out = config.get_output_path("pipeline_output") / "18n_driving_mini_causal_rule_reselection"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _cfg_key_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prediction_mode": str(cfg.get("prediction_mode", "any_rule_positive")),
        "top_k": int(cfg.get("top_k", 50)),
        "remove_pure_harmful": bool(cfg.get("remove_pure_harmful", True)),
        "downrank_redundant": bool(cfg.get("downrank_redundant", True)),
        "weak_grounding_penalty": float(cfg.get("weak_grounding_penalty", 1.5)),
        "harmful_penalty": float(cfg.get("harmful_penalty", 3.0)),
        "false_positive_penalty": float(cfg.get("false_positive_penalty", 2.0)),
        "necessary_positive_bonus": float(cfg.get("necessary_positive_bonus", 3.0)),
        "helpful_bonus": float(cfg.get("helpful_bonus", 1.0)),
        "object_support_bonus": float(cfg.get("object_support_bonus", 0.5)),
        "backup_explanation_min_trigger_count": int(cfg.get("backup_explanation_min_trigger_count", 5)),
        "backfill_mixed_rules": bool(cfg.get("backfill_mixed_rules", False)),
        "remove_mixed_harmful_dominant": bool(cfg.get("remove_mixed_harmful_dominant", False)),
        "skip_zero_positive_support_refill": bool(cfg.get("skip_zero_positive_support_refill", True)),
        "iterative_enabled": bool(cfg.get("iterative_enabled", False)),
        "max_rounds": int(cfg.get("max_rounds", 1)),
        "removal_mode": str(cfg.get("removal_mode", "strict")),
        "max_remove_fraction_per_round": float(cfg.get("max_remove_fraction_per_round", 0.2)),
        "selection_metric": str(cfg.get("selection_metric", "f1")),
        "tie_breaker": str(cfg.get("tie_breaker", "precision")),
        "early_stop_no_improvement_rounds": int(cfg.get("early_stop_no_improvement_rounds", 2)),
        "descendant_replacement_enabled": bool(cfg.get("descendant_replacement_enabled", False)),
        "max_replacement_parents_per_round": int(cfg.get("max_replacement_parents_per_round", 3)),
        "descendant_queue_size_per_parent": int(cfg.get("descendant_queue_size_per_parent", 10)),
        "max_descendant_replacements_per_round": int(cfg.get("max_descendant_replacements_per_round", 3)),
        "max_descendant_rounds": int(cfg.get("max_descendant_rounds", 6)),
        "broad_weak_predicates": sorted(str(value) for value in list(cfg.get("broad_weak_predicates", []))),
        "ego_motion_only_predicates": sorted(str(value) for value in list(cfg.get("ego_motion_only_predicates", []))),
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    return loaded if isinstance(loaded, dict) else {}


def _parse_atom(atom: str) -> Tuple[str, List[str]]:
    text = str(atom).strip().rstrip(".")
    match = re.match(r"^([A-Za-z0-9_]+)\((.*)\)$", text)
    if not match:
        return text, []
    args_text = match.group(2).strip()
    args = [part.strip() for part in args_text.split(",")] if args_text else []
    return match.group(1), args


def _split_clause_body_atoms(body: str) -> List[str]:
    atoms: List[str] = []
    current: List[str] = []
    depth = 0
    for char in str(body):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            atom = "".join(current).strip()
            if atom:
                atoms.append(atom)
            current = []
        else:
            current.append(char)
    atom = "".join(current).strip()
    if atom:
        atoms.append(atom)
    return atoms


def _body_atoms_from_clause(clause: str, *, trailing_dot: bool = False) -> List[str]:
    clause_text = str(clause or "").strip()
    if ":-" not in clause_text:
        return []
    body = clause_text.split(":-", 1)[1].strip()
    body = body.rsplit(".", 1)[0]
    atoms = _split_clause_body_atoms(body)
    if trailing_dot:
        return [atom if atom.endswith(".") else f"{atom}." for atom in atoms]
    return [atom.rstrip(".") for atom in atoms]


def _clause_head(clause: Any) -> str:
    text = str(clause or "").strip()
    if ":-" in text:
        text = text.split(":-", 1)[0].strip()
    return _normalize_clause(text)


def _normalized_body_atom_set(rule: Dict[str, Any]) -> set[str]:
    return {_normalize_clause(atom) for atom in _body_atoms(rule) if _normalize_clause(atom)}


def _normalize_clause(clause: Any) -> str:
    text = str(clause or "").strip().rstrip(".")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _body_atoms(rule: Dict[str, Any]) -> List[str]:
    for key in ("body_atom_templates", "body_atoms", "antecedent_atoms"):
        value = rule.get(key)
        if isinstance(value, list):
            atoms = [str(atom).strip() for atom in value if str(atom).strip()]
            if atoms:
                return atoms
    return _body_atoms_from_clause(str(rule.get("clause", "")))


def _rule_from_causal_row(causal: Dict[str, Any]) -> Dict[str, Any]:
    rule_id = str(causal.get("rule_id", "")).strip()
    clause = str(causal.get("clause", "")).strip()
    body_atom_templates = _body_atoms_from_clause(clause, trailing_dot=True)
    rule: Dict[str, Any] = {
        "rule_id": rule_id,
        "clause": clause,
        "confidence": _safe_float(causal.get("confidence", 0.0)),
        "score": _safe_float(causal.get("confidence", 0.0)),
        "body_atom_templates": body_atom_templates,
        "body_length": len(body_atom_templates),
        "selection_source": "step18m_backfill",
    }
    if body_atom_templates:
        rule["body_atom_template"] = body_atom_templates[0]
    return rule


def _rule_grounding_features(rule: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    atoms = _body_atoms(rule)
    broad_weak_predicates = {str(value) for value in list(cfg.get("broad_weak_predicates", []))}
    ego_motion_only_predicates = {str(value) for value in list(cfg.get("ego_motion_only_predicates", []))}
    predicates = []
    has_object_level_support = False
    uses_broad_weak_atoms = False
    for atom in atoms:
        predicate, args = _parse_atom(atom)
        predicates.append(predicate)
        atom_text = atom.lower()
        atom_uses_broad_weak = predicate in broad_weak_predicates or any(
            token.lower() in atom_text for token in broad_weak_predicates
        )
        if atom_uses_broad_weak:
            uses_broad_weak_atoms = True
        if any(arg == "O" or str(arg).startswith("object") for arg in args):
            if not atom_uses_broad_weak and "intermittent" not in atom_text:
                has_object_level_support = True
    uses_only_ego_motion_atoms = bool(atoms) and all(predicate in ego_motion_only_predicates for predicate in predicates)
    weak_causal_grounding = bool(uses_only_ego_motion_atoms or (uses_broad_weak_atoms and not has_object_level_support))
    return {
        "body_atoms": atoms,
        "has_object_level_support": has_object_level_support,
        "uses_only_ego_motion_atoms": uses_only_ego_motion_atoms,
        "uses_broad_weak_atoms": uses_broad_weak_atoms,
        "weak_causal_grounding": weak_causal_grounding,
    }


def _dominant_influence_indicates_redundancy(dominant_influence_type: str) -> bool:
    return str(dominant_influence_type).strip().lower() in {"redundant_trigger", "redundant_rule"}


def _has_nonzero_causal_effect(row: Dict[str, Any]) -> bool:
    causal_count_fields = (
        "prediction_flip_count",
        "helpful_count",
        "harmful_count",
        "necessary_true_positive_count",
        "causal_false_positive_count",
        "non_decisive_contribution_count",
    )
    if any(_safe_int(row.get(field, 0)) > 0 for field in causal_count_fields):
        return True
    return abs(_safe_float(row.get("score_delta_sum", 0.0), 0.0)) > 0.0


def _causal_effect_status(*, found_in_step18m: bool, causal: Dict[str, Any]) -> str:
    if not found_in_step18m:
        return "missing_from_step18m"
    if _has_nonzero_causal_effect(causal):
        return "nonzero_causal_effect"
    return "no_causal_effect"


def _is_refinement_target_candidate(grounding: Dict[str, Any]) -> bool:
    return bool(grounding.get("uses_broad_weak_atoms", False) or grounding.get("uses_only_ego_motion_atoms", False))


def _is_broad_high_coverage_rule(rule: Dict[str, Any], grounding: Dict[str, Any]) -> bool:
    atoms = list(grounding.get("body_atoms", [])) or _body_atoms(rule)
    if bool(grounding.get("uses_broad_weak_atoms", False) or grounding.get("uses_only_ego_motion_atoms", False)):
        return True
    if len(atoms) != 1:
        return False
    predicate, args = _parse_atom(str(atoms[0]))
    atom_text = str(atoms[0]).lower()
    if predicate == "object_distance_state" and any(str(arg).strip().lower() == "near" for arg in args):
        return True
    return "vz_approaching" in atom_text or "vz_awaying" in atom_text or "intermittent" in atom_text


def _row_brief(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule_id": str(row.get("rule_id", "")),
        "clause": str(row.get("original_clause", row.get("clause", ""))),
    }


def _load_ranked_rules(final_rule_results: Dict[str, Any], *, output_root: Path) -> List[Dict[str, Any]]:
    ranked_rules = [dict(rule) for rule in list(final_rule_results.get("ranked_rules", []))]
    if ranked_rules:
        return ranked_rules
    output_paths = dict(final_rule_results.get("output_paths", {}))
    ranked_path_text = str(output_paths.get("ranked_rules_json", ""))
    ranked_path = (
        Path(ranked_path_text)
        if ranked_path_text
        else output_root.parent / "17_driving_mini_final_rules" / "ranked_rules.json"
    )
    loaded = _load_json_if_exists(ranked_path)
    ranked_rules = [dict(rule) for rule in list(loaded.get("ranked_rules", []))]
    if ranked_rules:
        return ranked_rules
    return [dict(rule) for rule in list(final_rule_results.get("final_rules", []))]


def _rule_rank(rule: Dict[str, Any], default: int) -> int:
    return _safe_int(rule.get("original_rank", rule.get("rank", rule.get("selection_rank", default))), default)


def _rule_score(rule: Dict[str, Any]) -> float:
    return _safe_float(rule.get("original_score", rule.get("selection_utility", rule.get("confidence", 0.0))))


def _normalize_ranked_rule(rule: Dict[str, Any], rank: int) -> Dict[str, Any]:
    normalized = dict(rule)
    normalized.setdefault("rank", rank)
    normalized.setdefault("original_rank", _rule_rank(normalized, rank))
    normalized.setdefault("original_score", _rule_score(normalized))
    normalized.setdefault("confidence", _safe_float(normalized.get("confidence", normalized.get("original_score", 0.0))))
    normalized.setdefault("support", _safe_int(normalized.get("total_support", normalized.get("support", 0))))
    return normalized


def _rule_key(rule: Dict[str, Any]) -> str:
    rule_id = str(rule.get("rule_id", "")).strip()
    if rule_id:
        return f"id:{rule_id}"
    return f"clause:{_normalize_clause(rule.get('clause', ''))}"


def _load_rule_causal_rows(masking_results: Dict[str, Any], *, output_root: Path) -> Dict[str, Dict[str, Any]]:
    output_paths = dict(masking_results.get("output_paths", {}))
    path_text = str(output_paths.get("rule_causal_summary_csv", ""))
    path = (
        Path(path_text)
        if path_text
        else output_root.parent / "18m_driving_mini_rule_level_causal_masking" / "rule_causal_summary.csv"
    )
    rows = _read_csv_rows(path)
    causal_rows: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        rule_id = str(row.get("rule_id", "")).strip()
        if not rule_id:
            clause_key = _normalize_clause(row.get("clause", ""))
            rule_id = f"step18m_clause_{index:06d}" if clause_key else f"step18m_row_{index:06d}"
            row = {
                **dict(row),
                "rule_id": rule_id,
                "alignment_warning": "Step 18M causal summary row had no rule_id; using normalized clause matching.",
            }
        if rule_id in causal_rows:
            rule_id = f"{rule_id}__duplicate_{index:06d}"
            row = {**dict(row), "rule_id": rule_id}
        causal_rows[rule_id] = dict(row)
    return causal_rows


def _load_step18_rule_eval_rows(
    evaluation_results: Dict[str, Any],
    *,
    output_root: Path,
) -> Dict[str, Dict[str, Any]]:
    rows = list(evaluation_results.get("rule_evaluations", []))
    if rows:
        return {str(row.get("rule_id", "")): dict(row) for row in rows if str(row.get("rule_id", ""))}
    result_path = Path(str(evaluation_results.get("output_paths", {}).get("result_json", "")))
    if not result_path.exists():
        result_path = output_root.parent / "18_driving_mini_rule_evaluation" / "rule_evaluation.json"
    loaded = _load_json_if_exists(result_path)
    return {str(row.get("rule_id", "")): dict(row) for row in list(loaded.get("rule_evaluations", [])) if str(row.get("rule_id", ""))}


def _feedback_rule_counts(
    reasoning_feedback_results: Optional[Dict[str, Any]],
    *,
    output_root: Path,
) -> Dict[str, int]:
    if not reasoning_feedback_results:
        fallback = output_root.parent / "23_driving_mini_reasoning_feedback_signal" / "feedback_signal_summary.json"
        reasoning_feedback_results = _load_json_if_exists(fallback)
    counts: Dict[str, int] = defaultdict(int)
    for row in list((reasoning_feedback_results or {}).get("requests", [])):
        fired_rules = row.get("fired_rules_json", [])
        if isinstance(fired_rules, str):
            try:
                fired_rules = json.loads(fired_rules)
            except json.JSONDecodeError:
                fired_rules = []
        for rule in list(fired_rules or []):
            if isinstance(rule, dict):
                rule_id = str(rule.get("rule_id", "")).strip()
            else:
                rule_id = str(rule).strip()
            if rule_id:
                counts[rule_id] += 1
    return dict(counts)


def _classify_rule(
    *,
    found_in_step18m: bool,
    helpful_count: int,
    harmful_count: int,
    necessary_true_positive_count: int,
    causal_false_positive_count: int,
    prediction_flip_count: int,
    dominant_influence_type: str,
    weak_causal_grounding: bool,
) -> str:
    if not found_in_step18m:
        return "causal_effect_missing"
    if helpful_count == 0 and harmful_count > 0:
        return "harmful_false_positive_rule"
    if helpful_count > 0 and harmful_count > 0:
        return "mixed_rule"
    if necessary_true_positive_count > 0 and causal_false_positive_count > 0:
        return "mixed_rule"
    if necessary_true_positive_count > 0 and harmful_count == 0:
        return "necessary_positive_rule"
    if weak_causal_grounding:
        return "weak_causal_grounding_rule"
    if prediction_flip_count == 0 or _dominant_influence_indicates_redundancy(dominant_influence_type):
        return "redundant_rule"
    if helpful_count > 0:
        return "necessary_positive_rule"
    return "mixed_rule"


def _decision_for_category(
    category: str,
    *,
    found_in_step17: bool,
    helpful_count: int,
    harmful_count: int,
    necessary_true_positive_count: int,
    prediction_flip_count: int,
    trigger_count: int,
    found_in_step18m: bool,
    has_object_level_support: bool,
    refinement_target_candidate: bool,
    backup_min_trigger_count: int,
    remove_pure_harmful: bool,
    downrank_redundant: bool,
    backfill_mixed_rules: bool,
) -> Tuple[str, str, bool]:
    no_causal_effect = prediction_flip_count == 0 and helpful_count == 0 and harmful_count == 0
    if category == "causal_effect_missing":
        if trigger_count > 0 or has_object_level_support:
            return (
                "keep_for_review",
                "Step 17 rule was not found in Step 18M causal masking summary; causal effect is unknown",
                False,
            )
        return (
            "downrank_missing_effect",
            "Step 17 rule was not found in Step 18M and has no Step 18 trigger evidence",
            False,
        )
    if remove_pure_harmful and helpful_count == 0 and harmful_count > 0:
        return "remove", "masking only helped by removing this rule from false positives", False
    if category == "necessary_positive_rule":
        if not found_in_step17:
            return "backfill_keep", "Step 18M-only rule has necessary/helpful positive causal support", False
        if harmful_count == 0:
            return "keep", "masking shows necessary true-positive support without harmful flips", False
        return "keep", "rule preserves positive coverage despite some harmful contribution", False
    if category == "mixed_rule":
        if not found_in_step17:
            if backfill_mixed_rules:
                return (
                    "backfill_refine_candidate",
                    "Step 18M-only mixed rule allowed for low-priority backfill by config",
                    False,
                )
            return (
                "backfill_refine_candidate",
                "Step 18M-only mixed rule should be refined before direct backfill",
                False,
            )
        return "refine", "rule contributes to both true positives and false positives", False
    if category == "weak_causal_grounding_rule" and refinement_target_candidate:
        return "refine", "rule uses broad weak object-motion or ego-motion-only atoms with weak causal grounding", False
    if found_in_step18m and trigger_count == 0 and no_causal_effect:
        return "strong_downrank", "rule has zero Step 18M triggers and no measured causal effect", False
    if category == "redundant_rule" and downrank_redundant:
        if has_object_level_support and trigger_count >= backup_min_trigger_count:
            return (
                "backup_explanation",
                "redundant by masking but frequently triggered with object-level support",
                True,
            )
        if trigger_count == 0 and no_causal_effect:
            return "strong_downrank", "rule has zero trigger count and no measured causal effect", False
        return "downrank", "masking produced no final prediction change or dominant redundant influence", False
    if category == "weak_causal_grounding_rule" and necessary_true_positive_count <= 0:
        return "downrank", "rule depends on broad or ego-motion-only atoms without causal object support", False
    return "keep", "no causal evidence requiring removal", False


def _reselection_score(
    row: Dict[str, Any],
    cfg: Dict[str, Any],
) -> float:
    score = _safe_float(row.get("original_score", 0.0), 0.0)
    score += float(cfg.get("helpful_bonus", 1.0)) * _safe_int(row.get("helpful_count", 0))
    score += float(cfg.get("necessary_positive_bonus", 3.0)) * _safe_int(
        row.get("necessary_true_positive_count", 0)
    )
    score -= float(cfg.get("harmful_penalty", 3.0)) * _safe_int(row.get("harmful_count", 0))
    score -= float(cfg.get("false_positive_penalty", 2.0)) * _safe_int(
        row.get("causal_false_positive_count", 0)
    )
    if bool(row.get("has_object_level_support", False)):
        score += float(cfg.get("object_support_bonus", 0.5))
    if bool(row.get("weak_grounding_penalty_applied", False)):
        score -= float(cfg.get("weak_grounding_penalty", 1.5))
    if str(row.get("reselection_decision", "")) == "remove":
        score -= 1000000.0
    elif str(row.get("reselection_decision", "")) == "strong_downrank":
        score -= 100.0
    elif str(row.get("reselection_decision", "")) in {"downrank", "downrank_missing_effect"}:
        score -= 10.0
    elif str(row.get("reselection_decision", "")) == "keep_for_review":
        score -= 1.0
    elif str(row.get("reselection_decision", "")) == "backup_explanation":
        score -= 3.0
    elif str(row.get("reselection_decision", "")) == "backfill_keep":
        score += 0.25
    elif str(row.get("reselection_decision", "")) == "backfill_refine_candidate":
        score -= 20.0
    return score


def _make_reselection_row(
    *,
    rule: Dict[str, Any],
    causal: Dict[str, Any],
    eval_row: Dict[str, Any],
    rank: Any,
    found_in_step17: bool,
    found_in_step18m: bool,
    matched_by: str,
    alignment_warning: str,
    cfg: Dict[str, Any],
    feedback_counts: Dict[str, int],
    backup_min_trigger_count: int,
    backfill_mixed_rules: bool,
) -> Dict[str, Any]:
    rule_id = str(rule.get("rule_id", causal.get("rule_id", ""))).strip()
    clause = str(rule.get("clause", causal.get("clause", ""))).strip()
    helpful_count = _safe_int(causal.get("helpful_count", 0))
    harmful_count = _safe_int(causal.get("harmful_count", 0))
    necessary_true_positive_count = _safe_int(causal.get("necessary_true_positive_count", 0))
    causal_false_positive_count = _safe_int(causal.get("causal_false_positive_count", 0))
    redundant_count = _safe_int(causal.get("redundant_count", 0))
    prediction_flip_count = _safe_int(causal.get("prediction_flip_count", 0))
    trigger_count = _safe_int(causal.get("trigger_count", eval_row.get("eval_total_firings", 0)))
    dominant_influence_type = str(causal.get("dominant_influence_type", "none"))
    grounding = _rule_grounding_features(rule, cfg)
    causal_effect_status = _causal_effect_status(found_in_step18m=found_in_step18m, causal=causal)
    category = _classify_rule(
        found_in_step18m=found_in_step18m,
        helpful_count=helpful_count,
        harmful_count=harmful_count,
        necessary_true_positive_count=necessary_true_positive_count,
        causal_false_positive_count=causal_false_positive_count,
        prediction_flip_count=prediction_flip_count,
        dominant_influence_type=dominant_influence_type,
        weak_causal_grounding=bool(grounding["weak_causal_grounding"]),
    )
    decision, reason, backup_explanation_candidate = _decision_for_category(
        category,
        found_in_step17=found_in_step17,
        helpful_count=helpful_count,
        harmful_count=harmful_count,
        necessary_true_positive_count=necessary_true_positive_count,
        prediction_flip_count=prediction_flip_count,
        trigger_count=trigger_count,
        found_in_step18m=found_in_step18m,
        has_object_level_support=bool(grounding["has_object_level_support"]),
        refinement_target_candidate=_is_refinement_target_candidate(grounding),
        backup_min_trigger_count=backup_min_trigger_count,
        remove_pure_harmful=bool(cfg.get("remove_pure_harmful", True)),
        downrank_redundant=bool(cfg.get("downrank_redundant", True)),
        backfill_mixed_rules=backfill_mixed_rules,
    )
    if category == "weak_causal_grounding_rule" and feedback_counts.get(rule_id, 0) > 0 and decision != "remove":
        decision = "refine" if found_in_step17 else "backfill_refine_candidate"
        reason = f"{reason}; Step 23 feedback also flagged related examples"
    warning_message = ""
    if not found_in_step18m:
        warning_message = (
            "Step 17 rule_id was not found in Step 18M causal masking summary; "
            "causal effect statistics are missing, not zero."
        )
    elif alignment_warning:
        warning_message = alignment_warning
    selection_source = "original_step17" if found_in_step17 else "step18m_backfill"
    if decision == "backup_explanation":
        selection_source = "backup_explanation"
    elif decision == "keep_for_review":
        selection_source = "retained_review"
    backfill_reason = ""
    if not found_in_step17:
        backfill_reason = reason
        if decision == "backfill_refine_candidate" and not backfill_mixed_rules:
            backfill_reason = f"{reason}; backfill_mixed_rules=false"
        elif decision == "backfill_refine_candidate" and backfill_mixed_rules:
            backfill_reason = f"{reason}; backfill_mixed_rules=true"
    row = {
        "rule_id": rule_id,
        "clause": clause,
        "original_clause": clause,
        "original_rank": rank,
        "original_score": _safe_float(rule.get("score", rule.get("confidence", causal.get("confidence", 0.0)))),
        "found_in_step18m": bool(found_in_step18m),
        "missing_from_step18m": bool(not found_in_step18m),
        "found_in_step17": bool(found_in_step17),
        "matched_by": matched_by,
        "causal_effect_status": causal_effect_status,
        "alignment_warning": alignment_warning,
        "selection_source": selection_source,
        "backfill_reason": backfill_reason,
        "helpful_count": helpful_count,
        "harmful_count": harmful_count,
        "necessary_true_positive_count": necessary_true_positive_count,
        "causal_false_positive_count": causal_false_positive_count,
        "redundant_count": redundant_count,
        "prediction_flip_count": prediction_flip_count,
        "net_helpful_minus_harmful": helpful_count - harmful_count,
        "dominant_influence_type": dominant_influence_type,
        "assigned_reselection_category": category,
        "reselection_decision": decision,
        "reason": reason,
        "trigger_count": trigger_count,
        "non_decisive_contribution_count": _safe_int(causal.get("non_decisive_contribution_count", 0)),
        "weak_grounding_penalty_applied": bool(grounding["weak_causal_grounding"]),
        "has_object_level_support": bool(grounding["has_object_level_support"]),
        "uses_only_ego_motion_atoms": bool(grounding["uses_only_ego_motion_atoms"]),
        "uses_broad_weak_atoms": bool(grounding["uses_broad_weak_atoms"]),
        "reasoning_feedback_request_count": int(feedback_counts.get(rule_id, 0)),
        "backup_explanation_candidate": bool(backup_explanation_candidate),
        "warning_message": warning_message,
    }
    row["reselection_score"] = _reselection_score(row, cfg)
    return row


def _metric_value(evaluation_results: Dict[str, Any], metric: str) -> float:
    metrics = dict(evaluation_results.get("overall_metrics", {}))
    return _safe_float(metrics.get(metric, 0.0))


def _round_sort_key(
    row: Dict[str, Any],
    *,
    selection_metric: str,
    tie_breaker: str,
) -> Tuple[float, float, int, int]:
    metrics = dict(row.get("metrics", {}))
    return (
        _safe_float(metrics.get(selection_metric, 0.0)),
        _safe_float(metrics.get(tie_breaker, 0.0)),
        -_safe_int(row.get("num_cumulative_removed_rules", row.get("num_removed_rules", 0))),
        -_safe_int(row.get("round_index", 0)),
    )


def _is_better_round(
    candidate: Dict[str, Any],
    current_best: Dict[str, Any],
    *,
    selection_metric: str,
    tie_breaker: str,
) -> bool:
    if not current_best:
        return True
    return _round_sort_key(
        candidate,
        selection_metric=selection_metric,
        tie_breaker=tie_breaker,
    ) > _round_sort_key(
        current_best,
        selection_metric=selection_metric,
        tie_breaker=tie_breaker,
    )


def _can_use_refill_candidate(
    rule: Dict[str, Any],
    *,
    selected_rule_ids: set[str],
    selected_clause_keys: set[str],
    blacklisted_rule_ids: set[str],
    blacklisted_clauses: set[str],
    skip_zero_positive_support_refill: bool,
) -> bool:
    rule_id = str(rule.get("rule_id", "")).strip()
    clause_key = _normalize_clause(rule.get("clause", ""))
    if rule_id in selected_rule_ids or (clause_key and clause_key in selected_clause_keys):
        return False
    if rule_id in blacklisted_rule_ids or (clause_key and clause_key in blacklisted_clauses):
        return False
    if skip_zero_positive_support_refill and _safe_int(rule.get("positive_support", 0)) <= 0:
        return False
    return True


def _available_refill_count(
    ranked_rules: Sequence[Dict[str, Any]],
    current_rules: Sequence[Dict[str, Any]],
    *,
    blacklisted_rule_ids: set[str],
    blacklisted_clauses: set[str],
    skip_zero_positive_support_refill: bool,
) -> int:
    selected_rule_ids = {str(rule.get("rule_id", "")).strip() for rule in current_rules if str(rule.get("rule_id", "")).strip()}
    selected_clause_keys = {
        _normalize_clause(rule.get("clause", ""))
        for rule in current_rules
        if _normalize_clause(rule.get("clause", ""))
    }
    count = 0
    for rule in ranked_rules:
        if _can_use_refill_candidate(
            rule,
            selected_rule_ids=selected_rule_ids,
            selected_clause_keys=selected_clause_keys,
            blacklisted_rule_ids=blacklisted_rule_ids,
            blacklisted_clauses=blacklisted_clauses,
            skip_zero_positive_support_refill=skip_zero_positive_support_refill,
        ):
            count += 1
    return count


def _removal_reason_for_mode(
    row: Dict[str, Any],
    rule: Dict[str, Any],
    *,
    cfg: Dict[str, Any],
    removal_mode: str,
) -> str:
    helpful_count = _safe_int(row.get("helpful_count", 0))
    harmful_count = _safe_int(row.get("harmful_count", 0))
    necessary_true_positive_count = _safe_int(row.get("necessary_true_positive_count", 0))
    pure_harmful = helpful_count == 0 and harmful_count > 0 and necessary_true_positive_count == 0
    if pure_harmful:
        return "strict:pure harmful causal false-positive source"
    if removal_mode not in {"moderate", "aggressive"}:
        return ""
    if harmful_count >= 2 * max(1, helpful_count) and harmful_count > 0 and necessary_true_positive_count == 0:
        return "moderate:harmful-dominant mixed causal source"
    if removal_mode != "aggressive":
        return ""
    grounding = _rule_grounding_features(rule, cfg)
    eval_precision = _safe_float(row.get("eval_precision", rule.get("eval_precision", 0.0)))
    eval_fp_count = _safe_int(row.get("eval_fp_contribution_count_vs_accepted_only", rule.get("eval_fp_contribution_count_vs_accepted_only", 0)))
    broad_rule = _is_broad_high_coverage_rule(rule, grounding)
    weak_rule = bool(grounding.get("weak_causal_grounding", False))
    redundant_high_fp = (
        str(row.get("assigned_reselection_category", "")) == "redundant_rule"
        and eval_fp_count > 0
    )
    low_precision_threshold = _safe_float(cfg.get("aggressive_low_precision_threshold", 0.5), 0.5)
    if broad_rule and eval_precision < low_precision_threshold:
        return "aggressive:broad low-precision rule with replacement available"
    if redundant_high_fp:
        return "aggressive:redundant high-FP rule with replacement available"
    if weak_rule and necessary_true_positive_count == 0:
        return "aggressive:weak causal grounding rule with replacement available"
    return ""


def _candidate_removal_sort_key(row: Dict[str, Any]) -> Tuple[int, int, int, float, str]:
    reason = str(row.get("removal_reason", ""))
    mode_priority = 3 if reason.startswith("strict:") else 2 if reason.startswith("moderate:") else 1
    return (
        mode_priority,
        _safe_int(row.get("harmful_count", 0)) - _safe_int(row.get("helpful_count", 0)),
        _safe_int(row.get("causal_false_positive_count", 0)),
        _safe_float(row.get("eval_fp_contribution_rate_vs_accepted_only", 0.0)),
        str(row.get("rule_id", "")),
    )


def _is_descendant_rule(parent: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
    if _clause_head(parent.get("clause", "")) != _clause_head(candidate.get("clause", "")):
        return False
    parent_atoms = _normalized_body_atom_set(parent)
    candidate_atoms = _normalized_body_atom_set(candidate)
    return bool(parent_atoms) and parent_atoms < candidate_atoms


def _parent_replacement_reason(row: Dict[str, Any], rule: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    reasons: List[str] = []
    helpful_count = _safe_int(row.get("helpful_count", 0))
    harmful_count = _safe_int(row.get("harmful_count", 0))
    if str(row.get("assigned_reselection_category", "")) == "mixed_rule" and harmful_count > 0:
        reasons.append("mixed_harmful")
    atoms = _body_atoms(rule)
    if len(atoms) == 1:
        predicate, args = _parse_atom(str(atoms[0]))
        if any(arg == "O" or str(arg).startswith("object") for arg in args) or predicate.startswith("object_"):
            reasons.append("broad_one_atom_object_rule")
    if _safe_float(row.get("eval_precision", rule.get("eval_precision", 1.0)), 1.0) < _safe_float(
        cfg.get("descendant_low_precision_threshold", 0.5),
        0.5,
    ):
        reasons.append("low_precision")
    if _safe_int(rule.get("negative_support", 0)) > _safe_int(rule.get("positive_support", 0)):
        reasons.append("negative_support_gt_positive_support")
    if str(row.get("assigned_reselection_category", "")) == "weak_causal_grounding_rule" or bool(
        row.get("weak_grounding_penalty_applied", False)
    ):
        reasons.append("weak_causal_grounding")
    if _safe_int(row.get("eval_fp_contribution_count_vs_accepted_only", 0)) > 0 or _safe_int(
        row.get("causal_false_positive_count", 0)
    ) > 0:
        reasons.append("high_fp")
    return "|".join(sorted(set(reasons)))


def _descendant_score(
    *,
    parent_row: Dict[str, Any],
    parent_rule: Dict[str, Any],
    candidate: Dict[str, Any],
    candidate_rank: int,
    ranked_count: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    parent_precision = _safe_float(parent_row.get("eval_precision", parent_rule.get("eval_precision", 0.0)))
    candidate_precision = _safe_float(candidate.get("eval_precision", candidate.get("precision", candidate.get("confidence", 0.0))))
    precision_gain = candidate_precision - parent_precision
    parent_fp = _safe_int(
        parent_row.get(
            "eval_fp_contribution_count_vs_accepted_only",
            parent_rule.get("eval_fp_contribution_count_vs_accepted_only", parent_rule.get("negative_support", 0)),
        )
    )
    candidate_fp = _safe_int(candidate.get("eval_fp_contribution_count_vs_accepted_only", candidate.get("negative_support", 0)))
    fp_reduction = parent_fp - candidate_fp
    parent_pos = max(1, _safe_int(parent_rule.get("positive_support", parent_row.get("helpful_count", 0)), 1))
    candidate_pos = _safe_int(candidate.get("positive_support", 0))
    positive_coverage_retention = float(min(1.0, candidate_pos / max(1, parent_pos)))
    grounding = _rule_grounding_features(candidate, cfg)
    causal_grounding_bonus = 0.25 if bool(grounding.get("has_object_level_support", False)) else 0.0
    rank_bonus = float((ranked_count - candidate_rank + 1) / max(1, ranked_count))
    rule_length_penalty = 0.05 * max(0, len(_body_atoms(candidate)) - len(_body_atoms(parent_rule)))
    score = (
        2.0 * precision_gain
        + 1.0 * fp_reduction
        + 0.75 * positive_coverage_retention
        + causal_grounding_bonus
        + 0.25 * rank_bonus
        - rule_length_penalty
    )
    return {
        "score": score,
        "precision_gain": precision_gain,
        "fp_reduction": fp_reduction,
        "positive_coverage_retention": positive_coverage_retention,
        "causal_grounding_bonus": causal_grounding_bonus,
        "rank_bonus": rank_bonus,
        "rule_length_penalty": rule_length_penalty,
    }


def _build_descendant_replacement_plan(
    *,
    current_rules: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    ranked_rules: Sequence[Dict[str, Any]],
    blacklisted_rule_ids: set[str],
    blacklisted_clauses: set[str],
    failed_descendant_pairs: set[Tuple[str, str]],
    cfg: Dict[str, Any],
    round_index: int,
) -> Dict[str, Any]:
    if not bool(cfg.get("descendant_replacement_enabled", False)):
        return {"rules": [dict(rule) for rule in current_rules], "candidates": [], "trials": [], "summary": {}}
    max_parents = max(0, int(cfg.get("max_replacement_parents_per_round", 3)))
    queue_size = max(1, int(cfg.get("descendant_queue_size_per_parent", 10)))
    max_replacements = max(0, int(cfg.get("max_descendant_replacements_per_round", 3)))
    if max_parents <= 0 or max_replacements <= 0:
        return {"rules": [dict(rule) for rule in current_rules], "candidates": [], "trials": [], "summary": {}}

    rules_by_id = {str(rule.get("rule_id", "")).strip(): dict(rule) for rule in current_rules if str(rule.get("rule_id", "")).strip()}
    selected_rule_ids = set(rules_by_id)
    selected_clause_keys = {
        _normalize_clause(rule.get("clause", ""))
        for rule in current_rules
        if _normalize_clause(rule.get("clause", ""))
    }
    parent_rows: List[Dict[str, Any]] = []
    for row in selected_rows:
        rule_id = str(row.get("rule_id", "")).strip()
        rule = rules_by_id.get(rule_id)
        if not rule:
            continue
        reason = _parent_replacement_reason(row, rule, cfg)
        if reason:
            parent_row = dict(row)
            parent_row["parent_reason"] = reason
            parent_rows.append(parent_row)
    parent_rows = sorted(
        parent_rows,
        key=lambda row: (
            _safe_int(row.get("harmful_count", 0)) + _safe_int(row.get("causal_false_positive_count", 0)),
            _safe_int(row.get("eval_fp_contribution_count_vs_accepted_only", 0)),
            -_safe_float(row.get("eval_precision", 1.0), 1.0),
            str(row.get("rule_id", "")),
        ),
        reverse=True,
    )[:max_parents]

    candidate_rows: List[Dict[str, Any]] = []
    parent_queues: Dict[str, List[Dict[str, Any]]] = {}
    ranked_count = len(ranked_rules)
    rank_by_id = {str(rule.get("rule_id", "")).strip(): index for index, rule in enumerate(ranked_rules, start=1)}
    for parent_row in parent_rows:
        parent_id = str(parent_row.get("rule_id", "")).strip()
        parent_rule = rules_by_id.get(parent_id, {})
        queue: List[Dict[str, Any]] = []
        for candidate in ranked_rules:
            candidate_id = str(candidate.get("rule_id", "")).strip()
            clause_key = _normalize_clause(candidate.get("clause", ""))
            if (parent_id, candidate_id) in failed_descendant_pairs:
                continue
            if candidate_id in selected_rule_ids or (clause_key and clause_key in selected_clause_keys):
                continue
            if candidate_id in blacklisted_rule_ids or (clause_key and clause_key in blacklisted_clauses):
                continue
            if _safe_int(candidate.get("positive_support", 0)) <= 0:
                continue
            if not _is_descendant_rule(parent_rule, candidate):
                continue
            candidate_rank = rank_by_id.get(candidate_id, _rule_rank(candidate, ranked_count))
            score_parts = _descendant_score(
                parent_row=parent_row,
                parent_rule=parent_rule,
                candidate=candidate,
                candidate_rank=candidate_rank,
                ranked_count=ranked_count,
                cfg=cfg,
            )
            row = {
                "round_index": int(round_index),
                "parent_rule_id": parent_id,
                "parent_clause": str(parent_rule.get("clause", "")),
                "candidate_rule_id": candidate_id,
                "candidate_clause": str(candidate.get("clause", "")),
                "candidate_rank": candidate_rank,
                "parent_reason": str(parent_row.get("parent_reason", "")),
                **score_parts,
                "_candidate_rule": dict(candidate),
            }
            queue.append(row)
        queue = sorted(queue, key=lambda row: (_safe_float(row.get("score", 0.0)), -_safe_int(row.get("candidate_rank", 0))), reverse=True)[:queue_size]
        parent_queues[parent_id] = queue
        candidate_rows.extend(queue)

    used_parent_ids: set[str] = set()
    used_candidate_ids: set[str] = set()
    chosen: List[Dict[str, Any]] = []
    for candidate_row in sorted(candidate_rows, key=lambda row: (_safe_float(row.get("score", 0.0)), -_safe_int(row.get("candidate_rank", 0))), reverse=True):
        parent_id = str(candidate_row.get("parent_rule_id", ""))
        candidate_id = str(candidate_row.get("candidate_rule_id", ""))
        if parent_id in used_parent_ids or candidate_id in used_candidate_ids:
            continue
        chosen.append(candidate_row)
        used_parent_ids.add(parent_id)
        used_candidate_ids.add(candidate_id)
        if len(chosen) >= max_replacements:
            break

    replacements_by_parent = {str(row.get("parent_rule_id", "")): row for row in chosen}
    next_rules: List[Dict[str, Any]] = []
    trials: List[Dict[str, Any]] = []
    for rule in current_rules:
        parent_id = str(rule.get("rule_id", "")).strip()
        replacement = replacements_by_parent.get(parent_id)
        if not replacement:
            next_rules.append(dict(rule))
            continue
        descendant = dict(replacement.get("_candidate_rule", {}))
        descendant["selection_source"] = "descendant_replacement"
        descendant["replaced_parent_rule_id"] = parent_id
        next_rules.append(descendant)
        trials.append(
            {
                "round_index": int(round_index),
                "parent_rule_id": parent_id,
                "descendant_rule_id": str(descendant.get("rule_id", "")),
                "parent_clause": str(rule.get("clause", "")),
                "descendant_clause": str(descendant.get("clause", "")),
                "trial_status": "pending_validation",
                "validation_f1": "",
                "validation_precision": "",
                "improved_over_previous_best": False,
                "failure_reason": "",
            }
        )
    return {
        "rules": next_rules,
        "candidates": [{key: value for key, value in row.items() if key != "_candidate_rule"} for row in candidate_rows],
        "trials": trials,
        "summary": {
            "round_index": int(round_index),
            "num_candidate_parents": len(parent_rows),
            "num_candidate_descendants": len(candidate_rows),
            "num_trial_replacements": len(trials),
            "num_failed_trials": 0,
        },
    }


def _build_round_rule_results(
    *,
    rules: Sequence[Dict[str, Any]],
    base_results: Dict[str, Any],
    round_index: int,
) -> Dict[str, Any]:
    return {
        **dict(base_results),
        "final_rules": [dict(rule) for rule in rules],
        "num_final_rules": len(rules),
        "selection_method": "iterative_causal_aware_top_k_rule_maintenance",
        "iterative_round_index": int(round_index),
    }


def _write_refined_rules_artifacts(
    *,
    json_path: Path,
    csv_path: Path,
    result: Dict[str, Any],
    selected_rows: Sequence[Dict[str, Any]],
) -> None:
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    _write_csv(csv_path, _REFINED_RULE_FIELDS, selected_rows)


def _run_maintenance_pass(
    *,
    current_rules: Sequence[Dict[str, Any]],
    base_final_rule_results: Dict[str, Any],
    ranked_rules: Sequence[Dict[str, Any]],
    evaluation_results: Dict[str, Any],
    masking_results: Dict[str, Any],
    reasoning_feedback_results: Optional[Dict[str, Any]],
    cfg: Dict[str, Any],
    output_root: Path,
    top_k: int,
    round_index: int,
    blacklisted_rule_ids: set[str],
    blacklisted_clauses: set[str],
) -> Dict[str, Any]:
    causal_rows = _load_rule_causal_rows(masking_results, output_root=output_root)
    eval_rows = _load_step18_rule_eval_rows(evaluation_results, output_root=output_root)
    feedback_counts = _feedback_rule_counts(reasoning_feedback_results, output_root=output_root)
    backup_min_trigger_count = max(1, int(cfg.get("backup_explanation_min_trigger_count", 5)))
    skip_zero_positive_support_refill = bool(cfg.get("skip_zero_positive_support_refill", True))
    removal_mode = str(cfg.get("removal_mode", "strict")).strip().lower()
    if removal_mode not in {"strict", "moderate", "aggressive"}:
        removal_mode = "strict"
    max_remove_fraction = max(0.0, min(1.0, _safe_float(cfg.get("max_remove_fraction_per_round", 0.2), 0.2)))
    max_remove_count = int(max(1, int(top_k * max_remove_fraction))) if top_k > 0 and max_remove_fraction > 0 else 0

    active_rule_by_id = {str(rule.get("rule_id", "")).strip(): dict(rule) for rule in current_rules if str(rule.get("rule_id", "")).strip()}
    active_rule_by_clause = {
        _normalize_clause(rule.get("clause", "")): dict(rule)
        for rule in current_rules
        if _normalize_clause(rule.get("clause", ""))
    }
    consumed_causal_ids: set[str] = set()
    active_rows: List[Dict[str, Any]] = []
    refinement_target_rows: List[Dict[str, Any]] = []

    for active_rank, rule in enumerate(current_rules, start=1):
        rule_id = str(rule.get("rule_id", "")).strip()
        matched_by = "missing_from_step18m"
        alignment_warning = ""
        causal: Dict[str, Any] = {}
        causal_rule_id = ""
        if rule_id and rule_id in causal_rows:
            causal_rule_id = rule_id
            causal = dict(causal_rows[rule_id])
            matched_by = "rule_id"
        else:
            clause_key = _normalize_clause(rule.get("clause", ""))
            for candidate_causal_id, candidate_causal in causal_rows.items():
                if candidate_causal_id in consumed_causal_ids:
                    continue
                if clause_key and _normalize_clause(candidate_causal.get("clause", "")) == clause_key:
                    causal_rule_id = candidate_causal_id
                    causal = dict(candidate_causal)
                    matched_by = "normalized_clause"
                    alignment_warning = (
                        "Active rule_id was not found in Step 18M, but normalized clause matched "
                        f"Step 18M rule_id={candidate_causal_id}."
                    )
                    break
        if causal_rule_id:
            consumed_causal_ids.add(causal_rule_id)
        eval_row = dict(eval_rows.get(rule_id, {}))
        row = _make_reselection_row(
            rule=rule,
            causal=causal,
            eval_row=eval_row,
            rank=_rule_rank(rule, active_rank),
            found_in_step17=True,
            found_in_step18m=bool(causal),
            matched_by=matched_by,
            alignment_warning=alignment_warning,
            cfg=cfg,
            feedback_counts=feedback_counts,
            backup_min_trigger_count=backup_min_trigger_count,
            backfill_mixed_rules=False,
        )
        row["round_index"] = int(round_index)
        row["selection_source"] = str(rule.get("selection_source", row.get("selection_source", "original_topk_retained")))
        row["eval_precision"] = _safe_float(eval_row.get("eval_precision", rule.get("eval_precision", 0.0)))
        row["eval_fp_contribution_count_vs_accepted_only"] = _safe_int(
            eval_row.get(
                "eval_fp_contribution_count_vs_accepted_only",
                rule.get("eval_fp_contribution_count_vs_accepted_only", 0),
            )
        )
        row["eval_fp_contribution_rate_vs_accepted_only"] = _safe_float(
            eval_row.get(
                "eval_fp_contribution_rate_vs_accepted_only",
                rule.get("eval_fp_contribution_rate_vs_accepted_only", 0.0),
            )
        )
        helpful_count = _safe_int(row.get("helpful_count", 0))
        harmful_count = _safe_int(row.get("harmful_count", 0))
        mixed_rule = helpful_count > 0 and harmful_count > 0
        grounding = _rule_grounding_features(rule, cfg)
        broad_refinement_target = _is_broad_high_coverage_rule(rule, grounding)
        if mixed_rule:
            row["assigned_reselection_category"] = "mixed_rule"
            row["reselection_decision"] = "refine"
        elif broad_refinement_target and str(row.get("reselection_decision", "")) != "remove":
            row["assigned_reselection_category"] = (
                "weak_causal_grounding_rule"
                if bool(grounding.get("weak_causal_grounding", False))
                else str(row.get("assigned_reselection_category", "broad_high_coverage_rule"))
            )
            row["reselection_decision"] = "refine" if bool(grounding.get("weak_causal_grounding", False)) else "keep"
        row["removal_reason"] = _removal_reason_for_mode(row, rule, cfg=cfg, removal_mode=removal_mode)
        row["blacklist_status"] = ""
        row["refill_reason"] = ""
        row["replaced_removed_rule_id"] = ""
        if (
            str(row.get("assigned_reselection_category", "")) == "mixed_rule"
            or broad_refinement_target
            or str(row.get("assigned_reselection_category", "")) == "weak_causal_grounding_rule"
        ):
            row["suggested_action"] = (
                "refine_with_more_specific_descendant"
                if str(row.get("assigned_reselection_category", "")) in {"mixed_rule", "weak_causal_grounding_rule"}
                else "keep_but_monitor"
            )
            refinement_target_rows.append(row)
        active_rows.append(row)

    available_replacements = _available_refill_count(
        ranked_rules,
        current_rules,
        blacklisted_rule_ids=blacklisted_rule_ids,
        blacklisted_clauses=blacklisted_clauses,
        skip_zero_positive_support_refill=skip_zero_positive_support_refill,
    )
    removable_rows = sorted(
        [row for row in active_rows if str(row.get("removal_reason", ""))],
        key=_candidate_removal_sort_key,
        reverse=True,
    )
    remove_limit = min(len(removable_rows), max_remove_count, available_replacements)
    remove_ids = {str(row.get("rule_id", "")) for row in removable_rows[:remove_limit]}

    removed_rows: List[Dict[str, Any]] = []
    retained_rows: List[Dict[str, Any]] = []
    for row in active_rows:
        rule_id = str(row.get("rule_id", "")).strip()
        if rule_id in remove_ids:
            row["reselection_decision"] = "remove"
            row["blacklist_status"] = "blacklisted_harmful"
            removed_rows.append(row)
            if rule_id:
                blacklisted_rule_ids.add(rule_id)
            clause_key = _normalize_clause(row.get("clause", ""))
            if clause_key:
                blacklisted_clauses.add(clause_key)
        else:
            if str(row.get("reselection_decision", "")) == "remove":
                row["reselection_decision"] = "keep"
            if str(row.get("removal_reason", "")) and rule_id not in remove_ids:
                row["warning_message"] = (
                    f"{row.get('warning_message', '')}; removal candidate retained because per-round removal cap "
                    "or replacement availability was exhausted"
                ).strip("; ")
            retained_rows.append(row)

    selected_rules: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    selected_rule_ids: set[str] = set()
    selected_clause_keys: set[str] = set()
    for row in retained_rows:
        rule = dict(active_rule_by_id.get(str(row.get("rule_id", "")), {}))
        if not rule:
            rule = dict(active_rule_by_clause.get(_normalize_clause(row.get("clause", "")), {}))
        if not rule:
            continue
        rule["selection_source"] = str(row.get("selection_source", "original_topk_retained"))
        selected_rules.append(rule)
        selected_rows.append(row)
        selected_rule_ids.add(str(rule.get("rule_id", "")).strip())
        selected_clause_keys.add(_normalize_clause(rule.get("clause", "")))

    refilled_rows: List[Dict[str, Any]] = []
    scanned_for_refill = 0
    skipped_duplicate = 0
    skipped_blacklist = 0
    skipped_zero_positive_support = 0
    replacement_queue = [str(row.get("rule_id", "")) for row in removed_rows]
    for ranked_rule in ranked_rules:
        if len(selected_rules) >= top_k:
            break
        scanned_for_refill += 1
        rule_id = str(ranked_rule.get("rule_id", "")).strip()
        clause_key = _normalize_clause(ranked_rule.get("clause", ""))
        if rule_id in selected_rule_ids or (clause_key and clause_key in selected_clause_keys):
            skipped_duplicate += 1
            continue
        if rule_id in blacklisted_rule_ids or (clause_key and clause_key in blacklisted_clauses):
            skipped_blacklist += 1
            continue
        if skip_zero_positive_support_refill and _safe_int(ranked_rule.get("positive_support", 0)) <= 0:
            skipped_zero_positive_support += 1
            continue
        replaced_rule_id = replacement_queue.pop(0) if replacement_queue else ""
        refill_reason = "refill_removed_harmful_rule" if replaced_rule_id else "fill_to_top_k_budget"
        refill_rule = dict(ranked_rule)
        refill_rule["selection_source"] = "ranked_pool_refill"
        refill_row = {
            "round_index": int(round_index),
            "rule_id": rule_id,
            "clause": str(ranked_rule.get("clause", "")),
            "original_clause": str(ranked_rule.get("clause", "")),
            "original_rank": _rule_rank(ranked_rule, len(selected_rules) + 1),
            "original_score": _rule_score(ranked_rule),
            "found_in_step18m": False,
            "missing_from_step18m": True,
            "found_in_step17": True,
            "matched_by": "ranked_pool",
            "causal_effect_status": "not_evaluated_by_18m",
            "alignment_warning": "",
            "selection_source": "ranked_pool_refill",
            "backfill_reason": "",
            "helpful_count": 0,
            "harmful_count": 0,
            "necessary_true_positive_count": 0,
            "causal_false_positive_count": 0,
            "redundant_count": 0,
            "prediction_flip_count": 0,
            "net_helpful_minus_harmful": 0,
            "dominant_influence_type": "not_evaluated_by_18m",
            "assigned_reselection_category": "ranked_pool_refill_rule",
            "reselection_decision": "refill",
            "reason": refill_reason,
            "trigger_count": 0,
            "non_decisive_contribution_count": 0,
            "weak_grounding_penalty_applied": False,
            "has_object_level_support": False,
            "uses_only_ego_motion_atoms": False,
            "uses_broad_weak_atoms": False,
            "reasoning_feedback_request_count": 0,
            "backup_explanation_candidate": False,
            "warning_message": (
                "Step 17 ranked-pool refill rule was not evaluated by Step 18M in this iteration; "
                "causal effect statistics are missing, not zero."
            ),
            "reselection_score": _rule_score(ranked_rule),
            "blacklist_status": "",
            "removal_reason": "",
            "refill_reason": refill_reason,
            "refill_rank": len(refilled_rows) + 1,
            "replaced_removed_rule_id": replaced_rule_id,
        }
        refilled_rows.append(refill_row)
        selected_rows.append(refill_row)
        selected_rules.append(refill_rule)
        if rule_id:
            selected_rule_ids.add(rule_id)
        if clause_key:
            selected_clause_keys.add(clause_key)

    for new_rank, row in enumerate(selected_rows, start=1):
        row["refined_rank"] = new_rank
    refined_rules: List[Dict[str, Any]] = []
    for row, rule in zip(selected_rows, selected_rules):
        refined_rule = dict(rule)
        refined_rule["causal_reselection"] = {key: row.get(key, "") for key in _ROW_FIELDS}
        refined_rule["refined_rank"] = int(row.get("refined_rank", 0))
        refined_rules.append(refined_rule)

    return {
        "round_index": int(round_index),
        "final_rules": refined_rules,
        "selected_rows": selected_rows,
        "active_rows": active_rows,
        "removed_rows": removed_rows,
        "retained_rows": retained_rows,
        "refilled_rows": refilled_rows,
        "refinement_target_rows": refinement_target_rows,
        "causal_rows": causal_rows,
        "num_refill_skipped_duplicate": skipped_duplicate,
        "num_refill_skipped_blacklist": skipped_blacklist,
        "num_refill_skipped_zero_positive_support": skipped_zero_positive_support,
        "ranked_pool_candidates_scanned_for_refill": scanned_for_refill,
        "refined_final_rules_reached_top_k": len(refined_rules) == top_k,
        "available_replacements": available_replacements,
        "num_removal_candidates": len(removable_rows),
    }


def _process_iterative_reselection(
    *,
    final_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    rule_level_causal_masking_results: Dict[str, Any],
    reasoning_feedback_results: Optional[Dict[str, Any]],
    temporal_rule_results: Sequence[Dict[str, Any]],
    logic_atom_results: Sequence[Dict[str, Any]],
    eval_video_ids: Sequence[str],
    split_manifest: Dict[str, Any],
    cfg: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    from src.exp_driving_videos.modules import evaluate_rules_driving_mini
    from src.exp_driving_videos.modules import rule_level_causal_masking_driving_mini

    result_path = output_root / "causal_rule_reselection.json"
    refined_json_path = output_root / "refined_final_rules.json"
    refined_csv_path = output_root / "refined_final_rules.csv"
    summary_csv_path = output_root / "causal_rule_reselection_summary.csv"
    removed_csv_path = output_root / "removed_rules.csv"
    retained_csv_path = output_root / "retained_rules.csv"
    refilled_csv_path = output_root / "refilled_rules.csv"
    refinement_targets_csv_path = output_root / "refinement_targets.csv"
    per_round_evaluation_csv_path = output_root / "per_round_rule_evaluation.csv"
    per_round_removed_csv_path = output_root / "per_round_removed_rules.csv"
    per_round_refilled_csv_path = output_root / "per_round_refilled_rules.csv"
    iterative_summary_csv_path = output_root / "iterative_reselection_summary.csv"
    best_round_refined_csv_path = output_root / "best_round_refined_final_rules.csv"
    best_round_manifest_path = output_root / "best_round_manifest.json"
    descendant_candidates_csv_path = output_root / "descendant_replacement_candidates.csv"
    descendant_trials_csv_path = output_root / "descendant_replacement_trials.csv"
    descendant_round_summary_csv_path = output_root / "per_round_descendant_replacement_summary.csv"
    failed_descendant_trials_csv_path = output_root / "failed_descendant_trials.csv"

    top_k = max(0, int(cfg.get("top_k", len(final_rule_results.get("final_rules", [])) or 50)))
    max_rounds = max(1, int(cfg.get("max_rounds", 5)))
    if bool(cfg.get("descendant_replacement_enabled", False)):
        max_rounds = max(max_rounds, max(1, int(cfg.get("max_descendant_rounds", 6))))
    selection_metric = str(cfg.get("selection_metric", "f1")).strip() or "f1"
    tie_breaker = str(cfg.get("tie_breaker", "precision")).strip() or "precision"
    early_stop_limit = max(0, int(cfg.get("early_stop_no_improvement_rounds", 2)))
    ranked_rules = [
        _normalize_ranked_rule(rule, rank)
        for rank, rule in enumerate(_load_ranked_rules(final_rule_results, output_root=output_root), start=1)
    ]
    current_rules = [dict(rule) for rule in list(final_rule_results.get("final_rules", []))]
    blacklisted_rule_ids: set[str] = set()
    blacklisted_clauses: set[str] = set()
    round_records: List[Dict[str, Any]] = []
    all_summary_rows: List[Dict[str, Any]] = []
    all_removed_rows: List[Dict[str, Any]] = []
    all_retained_rows: List[Dict[str, Any]] = []
    all_refilled_rows: List[Dict[str, Any]] = []
    all_refinement_target_rows: List[Dict[str, Any]] = []
    all_descendant_candidate_rows: List[Dict[str, Any]] = []
    all_descendant_trial_rows: List[Dict[str, Any]] = []
    all_descendant_round_rows: List[Dict[str, Any]] = []
    failed_descendant_trial_rows: List[Dict[str, Any]] = []
    failed_descendant_pairs: set[Tuple[str, str]] = set()

    baseline_metrics = dict(evaluation_results.get("overall_metrics", {}))
    best_record: Dict[str, Any] = {
        "round_index": 0,
        "rule_set_name": "round_00_initial",
        "final_rules": [dict(rule) for rule in current_rules],
        "selected_rows": [],
        "metrics": baseline_metrics,
        "evaluation_results": dict(evaluation_results),
        "num_removed_rules": 0,
        "num_cumulative_removed_rules": 0,
        "num_refilled_rules": 0,
        "num_blacklisted_rules": 0,
        "top_k_reached": len(current_rules) == top_k,
    }
    round_records.append(best_record)
    no_improvement_rounds = 0
    current_eval = dict(evaluation_results)
    current_masking = dict(rule_level_causal_masking_results)
    stop_reason = ""

    for round_index in range(1, max_rounds + 1):
        round_root = output_root / f"round_{round_index:02d}"
        round_root.mkdir(parents=True, exist_ok=True)
        maintenance = _run_maintenance_pass(
            current_rules=current_rules,
            base_final_rule_results=final_rule_results,
            ranked_rules=ranked_rules,
            evaluation_results=current_eval,
            masking_results=current_masking,
            reasoning_feedback_results=reasoning_feedback_results,
            cfg=cfg,
            output_root=round_root,
            top_k=top_k,
            round_index=round_index,
            blacklisted_rule_ids=blacklisted_rule_ids,
            blacklisted_clauses=blacklisted_clauses,
        )
        maintenance_rules = [dict(rule) for rule in list(maintenance.get("final_rules", []))]
        descendant_plan = _build_descendant_replacement_plan(
            current_rules=maintenance_rules,
            selected_rows=list(maintenance.get("selected_rows", [])),
            ranked_rules=ranked_rules,
            blacklisted_rule_ids=blacklisted_rule_ids,
            blacklisted_clauses=blacklisted_clauses,
            failed_descendant_pairs=failed_descendant_pairs,
            cfg=cfg,
            round_index=round_index,
        )
        next_rules = [dict(rule) for rule in list(descendant_plan.get("rules", maintenance_rules))]
        descendant_trials = [dict(row) for row in list(descendant_plan.get("trials", []))]
        descendant_candidates = [dict(row) for row in list(descendant_plan.get("candidates", []))]
        selected_rows_for_round = list(maintenance.get("selected_rows", []))
        if descendant_trials:
            trial_by_parent = {str(row.get("parent_rule_id", "")): row for row in descendant_trials}
            replaced_rows: List[Dict[str, Any]] = []
            for row in selected_rows_for_round:
                parent_id = str(row.get("rule_id", "")).strip()
                trial = trial_by_parent.get(parent_id)
                if not trial:
                    replaced_rows.append(row)
                    continue
                descendant_rule = next(
                    (rule for rule in next_rules if str(rule.get("rule_id", "")) == str(trial.get("descendant_rule_id", ""))),
                    {},
                )
                replacement_row = dict(row)
                replacement_row.update(
                    {
                        "rule_id": str(descendant_rule.get("rule_id", "")),
                        "clause": str(descendant_rule.get("clause", "")),
                        "selection_source": "descendant_replacement",
                        "reselection_decision": "replace_with_descendant",
                        "assigned_reselection_category": "descendant_replacement_rule",
                        "found_in_step18m": False,
                        "missing_from_step18m": True,
                        "causal_effect_status": "not_evaluated_by_18m",
                        "warning_message": "Descendant replacement trial; causal effect measured after this round evaluation.",
                        "refill_reason": "descendant_replacement",
                        "replaced_removed_rule_id": parent_id,
                        "blacklist_status": "",
                    }
                )
                replaced_rows.append(replacement_row)
            selected_rows_for_round = replaced_rows
        round_rule_results = _build_round_rule_results(
            rules=next_rules,
            base_results=final_rule_results,
            round_index=round_index,
        )
        round_rule_results["output_paths"] = {
            **dict(final_rule_results.get("output_paths", {})),
            "refined_final_rules_json": str(round_root / "refined_final_rules.json"),
            "refined_final_rules_csv": str(round_root / "refined_final_rules.csv"),
        }
        round_rule_results["causal_rule_reselection_round"] = {
            "round_index": round_index,
            "removed_rules": list(maintenance.get("removed_rows", [])),
            "refilled_rules": list(maintenance.get("refilled_rows", [])),
            "descendant_replacement_trials": descendant_trials,
        }
        _write_refined_rules_artifacts(
            json_path=round_root / "refined_final_rules.json",
            csv_path=round_root / "refined_final_rules.csv",
            result=round_rule_results,
            selected_rows=selected_rows_for_round,
        )
        _write_csv(round_root / "causal_rule_reselection_summary.csv", _ROW_FIELDS, list(maintenance.get("active_rows", [])) + list(maintenance.get("refilled_rows", [])))
        _write_csv(round_root / "removed_rules.csv", _REMOVED_RULE_FIELDS, list(maintenance.get("removed_rows", [])))
        _write_csv(round_root / "refilled_rules.csv", _REFILLED_RULE_FIELDS, list(maintenance.get("refilled_rows", [])))
        _write_csv(round_root / "refinement_targets.csv", _REFINEMENT_TARGET_FIELDS, list(maintenance.get("refinement_target_rows", [])))

        round_eval_root = round_root / "evaluation"
        round_evaluation = evaluate_rules_driving_mini.run(
            final_rule_results=round_rule_results,
            temporal_rule_results=list(temporal_rule_results),
            logic_atom_results=list(logic_atom_results),
            eval_video_ids=list(eval_video_ids),
            split_manifest=split_manifest,
            cfg={
                "prediction_mode": str(cfg.get("prediction_mode", "any_rule_positive")),
                "evaluated_rule_set_name": f"iterative_round_{round_index:02d}",
            },
            output_root=round_eval_root,
            force_recompute=True,
        )
        round_masking_root = round_root / "masking"
        round_masking = rule_level_causal_masking_driving_mini.run(
            final_rule_results=round_rule_results,
            evaluation_results=round_evaluation,
            cfg={
                "prediction_mode": str(cfg.get("prediction_mode", "any_rule_positive")),
                "rule_set_name": f"iterative_round_{round_index:02d}",
                "score_epsilon": float(cfg.get("score_epsilon", 1e-9)),
                "many_redundant_rule_fraction": float(cfg.get("many_redundant_rule_fraction", 0.5)),
            },
            output_root=round_masking_root,
            force_recompute=True,
        )

        record = {
            "round_index": round_index,
            "rule_set_name": f"round_{round_index:02d}",
            "final_rules": next_rules,
            "selected_rows": selected_rows_for_round,
            "metrics": dict(round_evaluation.get("overall_metrics", {})),
            "evaluation_results": round_evaluation,
            "masking_results": round_masking,
            "num_removed_rules": len(list(maintenance.get("removed_rows", []))),
            "num_cumulative_removed_rules": len(all_removed_rows) + len(list(maintenance.get("removed_rows", []))),
            "num_refilled_rules": len(list(maintenance.get("refilled_rows", []))),
            "num_blacklisted_rules": len(blacklisted_rule_ids),
            "top_k_reached": len(next_rules) == top_k,
            "removed_rules": list(maintenance.get("removed_rows", [])),
            "refilled_rules": list(maintenance.get("refilled_rows", [])),
            "retained_rules": list(maintenance.get("retained_rows", [])),
            "refinement_targets": list(maintenance.get("refinement_target_rows", [])),
            "descendant_replacement_candidates": descendant_candidates,
            "descendant_replacement_trials": descendant_trials,
            "ranked_pool_candidates_scanned_for_refill": int(
                maintenance.get("ranked_pool_candidates_scanned_for_refill", 0)
            ),
            "num_refill_skipped_duplicate": int(maintenance.get("num_refill_skipped_duplicate", 0)),
            "num_refill_skipped_blacklist": int(maintenance.get("num_refill_skipped_blacklist", 0)),
            "num_refill_skipped_zero_positive_support": int(
                maintenance.get("num_refill_skipped_zero_positive_support", 0)
            ),
            "round_output_root": str(round_root),
        }
        improved_round = _is_better_round(record, best_record, selection_metric=selection_metric, tie_breaker=tie_breaker)
        for trial in descendant_trials:
            trial["validation_f1"] = _safe_float(record["metrics"].get("f1", 0.0))
            trial["validation_precision"] = _safe_float(record["metrics"].get("precision", 0.0))
            trial["improved_over_previous_best"] = bool(improved_round)
            if not improved_round:
                trial["trial_status"] = "failed_trial"
                trial["failure_reason"] = "did_not_improve_validation_selection_metric"
                failed_descendant_pairs.add((str(trial.get("parent_rule_id", "")), str(trial.get("descendant_rule_id", ""))))
                failed_descendant_trial_rows.append(dict(trial))
            else:
                trial["trial_status"] = "accepted_trial"
        descendant_round_summary = dict(descendant_plan.get("summary", {}))
        descendant_round_summary.update(
            {
                "validation_f1": _safe_float(record["metrics"].get("f1", 0.0)),
                "validation_precision": _safe_float(record["metrics"].get("precision", 0.0)),
                "selected_as_best": bool(improved_round),
                "num_failed_trials": sum(1 for trial in descendant_trials if str(trial.get("trial_status", "")) == "failed_trial"),
            }
        )
        round_records.append(record)
        all_summary_rows.extend(list(maintenance.get("active_rows", [])) + list(maintenance.get("refilled_rows", [])))
        all_removed_rows.extend(list(maintenance.get("removed_rows", [])))
        all_retained_rows.extend(list(maintenance.get("retained_rows", [])))
        all_refilled_rows.extend(list(maintenance.get("refilled_rows", [])))
        all_refinement_target_rows.extend(list(maintenance.get("refinement_target_rows", [])))
        all_descendant_candidate_rows.extend(descendant_candidates)
        all_descendant_trial_rows.extend(descendant_trials)
        all_descendant_round_rows.append(descendant_round_summary)

        if improved_round:
            best_record = record
            no_improvement_rounds = 0
        else:
            no_improvement_rounds += 1
        if not record["removed_rules"] and not record["refilled_rules"] and not descendant_trials:
            stop_reason = "no_rules_removed_or_refilled"
            break
        if early_stop_limit and no_improvement_rounds >= early_stop_limit:
            stop_reason = f"no_validation_improvement_for_{early_stop_limit}_rounds"
            break
        if improved_round or not descendant_trials:
            current_rules = next_rules
            current_eval = round_evaluation
            current_masking = round_masking

    for record in round_records:
        record["selected_as_best"] = int(record.get("round_index", -1)) == int(best_record.get("round_index", -1))
        record["early_stop_reason"] = stop_reason if record is round_records[-1] else ""

    best_rules = [dict(rule) for rule in list(best_record.get("final_rules", []))]
    best_rows = list(best_record.get("selected_rows", []))
    if not best_rows:
        best_rows = [
            {
                "rule_id": str(rule.get("rule_id", "")),
                "clause": str(rule.get("clause", "")),
                "original_rank": _rule_rank(rule, index),
                "original_score": _rule_score(rule),
                "selection_source": "original_topk_retained",
                "reselection_decision": "keep",
                "assigned_reselection_category": "initial_rule",
                "helpful_count": 0,
                "harmful_count": 0,
                "necessary_true_positive_count": 0,
                "causal_false_positive_count": 0,
                "prediction_flip_count": 0,
                "found_in_step18m": False,
                "blacklist_status": "",
                "refill_reason": "",
                "replaced_removed_rule_id": "",
                "refined_rank": index,
            }
            for index, rule in enumerate(best_rules, start=1)
        ]

    per_round_rows: List[Dict[str, Any]] = []
    for record in round_records:
        metrics = dict(record.get("metrics", {}))
        per_round_rows.append(
            {
                "round_index": int(record.get("round_index", 0)),
                "rule_set_name": str(record.get("rule_set_name", "")),
                "num_rules": len(list(record.get("final_rules", []))),
                "num_removed_rules": int(record.get("num_removed_rules", 0)),
                "num_refilled_rules": int(record.get("num_refilled_rules", 0)),
                "num_blacklisted_rules": int(record.get("num_blacklisted_rules", 0)),
                "top_k_reached": bool(record.get("top_k_reached", False)),
                "true_positive": _safe_int(metrics.get("true_positive", 0)),
                "false_positive": _safe_int(metrics.get("false_positive", 0)),
                "false_negative": _safe_int(metrics.get("false_negative", 0)),
                "true_negative": _safe_int(metrics.get("true_negative", 0)),
                "precision": _safe_float(metrics.get("precision", 0.0)),
                "recall": _safe_float(metrics.get("recall", 0.0)),
                "f1": _safe_float(metrics.get("f1", 0.0)),
                "accuracy": _safe_float(metrics.get("accuracy", 0.0)),
                "selected_as_best": bool(record.get("selected_as_best", False)),
                "early_stop_reason": str(record.get("early_stop_reason", "")),
            }
        )

    _write_csv(summary_csv_path, _ROW_FIELDS, all_summary_rows)
    _write_csv(removed_csv_path, _REMOVED_RULE_FIELDS, all_removed_rows)
    _write_csv(retained_csv_path, _ROW_FIELDS, all_retained_rows)
    _write_csv(refilled_csv_path, _REFILLED_RULE_FIELDS, all_refilled_rows)
    _write_csv(refinement_targets_csv_path, _REFINEMENT_TARGET_FIELDS, all_refinement_target_rows)
    _write_csv(per_round_evaluation_csv_path, _PER_ROUND_EVALUATION_FIELDS, per_round_rows)
    _write_csv(per_round_removed_csv_path, ["round_index", *_REMOVED_RULE_FIELDS], all_removed_rows)
    _write_csv(per_round_refilled_csv_path, ["round_index", *_REFILLED_RULE_FIELDS], all_refilled_rows)
    _write_csv(iterative_summary_csv_path, _PER_ROUND_EVALUATION_FIELDS, per_round_rows)
    _write_csv(refined_csv_path, _REFINED_RULE_FIELDS, best_rows)
    _write_csv(best_round_refined_csv_path, _REFINED_RULE_FIELDS, best_rows)
    _write_csv(descendant_candidates_csv_path, _DESCENDANT_CANDIDATE_FIELDS, all_descendant_candidate_rows)
    _write_csv(descendant_trials_csv_path, _DESCENDANT_TRIAL_FIELDS, all_descendant_trial_rows)
    _write_csv(descendant_round_summary_csv_path, _DESCENDANT_ROUND_SUMMARY_FIELDS, all_descendant_round_rows)
    _write_csv(failed_descendant_trials_csv_path, _DESCENDANT_TRIAL_FIELDS, failed_descendant_trial_rows)

    best_round_manifest = {
        "best_round_index": int(best_record.get("round_index", 0)),
        "selection_metric": selection_metric,
        "tie_breaker": tie_breaker,
        "selected_on_validation_performance": True,
        "metrics": dict(best_record.get("metrics", {})),
        "num_rules": len(best_rules),
        "num_removed_rules": int(best_record.get("num_removed_rules", 0)),
        "num_refilled_rules": int(best_record.get("num_refilled_rules", 0)),
    }
    with best_round_manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(best_round_manifest, fh, indent=2)

    refined_result = {
        "version": _RESELECTION_VERSION,
        "config": _cfg_key_subset(cfg),
        "selection_method": "iterative_causal_aware_top_k_rule_maintenance",
        "selection_input_stage": "step17_ranked_pool_plus_iterative_step18_evaluation_and_step18m_causal_masking",
        "iterative_enabled": True,
        "top_k": top_k,
        "max_rounds": max_rounds,
        "num_rounds_executed": max(0, len(round_records) - 1),
        "best_round_index": int(best_record.get("round_index", 0)),
        "best_round_metrics": dict(best_record.get("metrics", {})),
        "selected_on_validation_performance": True,
        "selection_metric": selection_metric,
        "tie_breaker": tie_breaker,
        "early_stop_reason": stop_reason,
        "num_input_rules": len(final_rule_results.get("final_rules", [])),
        "num_initial_final_rules": len(final_rule_results.get("final_rules", [])),
        "num_final_rules": len(best_rules),
        "num_refined_final_rules": len(best_rules),
        "num_removed_rules": len(all_removed_rows),
        "num_removed_harmful_rules": len(all_removed_rows),
        "num_refilled_rules": len(all_refilled_rows),
        "num_blacklisted_rules": len(blacklisted_rule_ids),
        "num_refinement_targets": len(all_refinement_target_rows),
        "descendant_replacement_enabled": bool(cfg.get("descendant_replacement_enabled", False)),
        "num_descendant_replacement_candidates": len(all_descendant_candidate_rows),
        "num_descendant_replacement_trials": len(all_descendant_trial_rows),
        "num_failed_descendant_trials": len(failed_descendant_trial_rows),
        "refined_final_rules_reached_top_k": len(best_rules) == top_k,
        "rounds": round_records,
        "per_round_evaluation_rows": per_round_rows,
        "removed_rules": all_removed_rows,
        "refilled_rules": all_refilled_rows,
        "refinement_targets": all_refinement_target_rows,
        "descendant_replacement_candidates": all_descendant_candidate_rows,
        "descendant_replacement_trials": all_descendant_trial_rows,
        "failed_descendant_trials": failed_descendant_trial_rows,
        "final_rules": best_rules,
        "reselection_rows": all_summary_rows,
        "output_paths": {
            "result_json": str(result_path),
            "refined_final_rules_json": str(refined_json_path),
            "refined_final_rules_csv": str(refined_csv_path),
            "causal_rule_reselection_summary_csv": str(summary_csv_path),
            "removed_rules_csv": str(removed_csv_path),
            "retained_rules_csv": str(retained_csv_path),
            "refilled_rules_csv": str(refilled_csv_path),
            "refinement_targets_csv": str(refinement_targets_csv_path),
            "per_round_rule_evaluation_csv": str(per_round_evaluation_csv_path),
            "per_round_removed_rules_csv": str(per_round_removed_csv_path),
            "per_round_refilled_rules_csv": str(per_round_refilled_csv_path),
            "iterative_reselection_summary_csv": str(iterative_summary_csv_path),
            "best_round_refined_final_rules_csv": str(best_round_refined_csv_path),
            "best_round_manifest_json": str(best_round_manifest_path),
            "descendant_replacement_candidates_csv": str(descendant_candidates_csv_path),
            "descendant_replacement_trials_csv": str(descendant_trials_csv_path),
            "per_round_descendant_replacement_summary_csv": str(descendant_round_summary_csv_path),
            "failed_descendant_trials_csv": str(failed_descendant_trials_csv_path),
        },
    }
    with refined_json_path.open("w", encoding="utf-8") as fh:
        json.dump(refined_result, fh, indent=2)
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(refined_result, fh, indent=2)
    print(
        "  iterative_causal_rule_reselection: "
        f"rounds={refined_result['num_rounds_executed']} | "
        f"best_round={refined_result['best_round_index']} | "
        f"best_f1={_safe_float(refined_result['best_round_metrics'].get('f1', 0.0)):.3f} | "
        f"rules={len(best_rules)}"
    )
    refined_result["_freshly_computed"] = True
    return refined_result


def process_reselection(
    final_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    rule_level_causal_masking_results: Dict[str, Any],
    reasoning_feedback_results: Optional[Dict[str, Any]] = None,
    temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    logic_atom_results: Optional[Sequence[Dict[str, Any]]] = None,
    eval_video_ids: Optional[Sequence[str]] = None,
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or {}
    out_root = output_root or get_output_root()
    out_root.mkdir(parents=True, exist_ok=True)
    result_path = out_root / "causal_rule_reselection.json"
    refined_json_path = out_root / "refined_final_rules.json"
    refined_csv_path = out_root / "refined_final_rules.csv"
    summary_csv_path = out_root / "causal_rule_reselection_summary.csv"
    removed_csv_path = out_root / "removed_rules.csv"
    retained_csv_path = out_root / "retained_rules.csv"
    refilled_csv_path = out_root / "refilled_rules.csv"
    refinement_targets_csv_path = out_root / "refinement_targets.csv"

    cfg_subset = _cfg_key_subset(cfg)
    if not force_recompute and result_path.exists():
        cached = _load_json_if_exists(result_path)
        if int(cached.get("version", 0)) == _RESELECTION_VERSION and dict(cached.get("config", {})) == cfg_subset:
            print(f"  [cache] loading {result_path.name}")
            cached["_freshly_computed"] = False
            return cached

    if bool(cfg.get("iterative_enabled", False)) and temporal_rule_results is not None:
        return _process_iterative_reselection(
            final_rule_results=final_rule_results,
            evaluation_results=evaluation_results,
            rule_level_causal_masking_results=rule_level_causal_masking_results,
            reasoning_feedback_results=reasoning_feedback_results,
            temporal_rule_results=list(temporal_rule_results or []),
            logic_atom_results=list(logic_atom_results or []),
            eval_video_ids=list(eval_video_ids or []),
            split_manifest=dict(split_manifest or {}),
            cfg=cfg,
            output_root=out_root,
        )

    final_rules = [dict(rule) for rule in list(final_rule_results.get("final_rules", []))]
    causal_rows = _load_rule_causal_rows(rule_level_causal_masking_results, output_root=out_root)
    eval_rows = _load_step18_rule_eval_rows(evaluation_results, output_root=out_root)
    feedback_counts = _feedback_rule_counts(reasoning_feedback_results, output_root=out_root)
    top_k = max(0, int(cfg.get("top_k", len(final_rules) or 50)))
    backup_min_trigger_count = max(1, int(cfg.get("backup_explanation_min_trigger_count", 5)))
    remove_mixed_harmful_dominant = bool(cfg.get("remove_mixed_harmful_dominant", False))
    skip_zero_positive_support_refill = bool(cfg.get("skip_zero_positive_support_refill", True))
    ranked_rules = [
        _normalize_ranked_rule(rule, rank)
        for rank, rule in enumerate(_load_ranked_rules(final_rule_results, output_root=out_root), start=1)
    ]

    active_rule_by_id = {str(rule.get("rule_id", "")).strip(): dict(rule) for rule in final_rules if str(rule.get("rule_id", "")).strip()}
    active_rule_by_clause = {
        _normalize_clause(rule.get("clause", "")): dict(rule)
        for rule in final_rules
        if _normalize_clause(rule.get("clause", ""))
    }
    consumed_causal_ids: set[str] = set()
    active_rows: List[Dict[str, Any]] = []
    removed_rows: List[Dict[str, Any]] = []
    retained_rows: List[Dict[str, Any]] = []
    refinement_target_rows: List[Dict[str, Any]] = []
    blacklisted_rule_ids: set[str] = set()
    blacklisted_clauses: set[str] = set()

    for active_rank, rule in enumerate(final_rules, start=1):
        rule_id = str(rule.get("rule_id", "")).strip()
        matched_by = "missing_from_step18m"
        alignment_warning = ""
        causal: Dict[str, Any] = {}
        causal_rule_id = ""
        if rule_id and rule_id in causal_rows:
            causal_rule_id = rule_id
            causal = dict(causal_rows[rule_id])
            matched_by = "rule_id"
        else:
            clause_key = _normalize_clause(rule.get("clause", ""))
            for candidate_causal_id, candidate_causal in causal_rows.items():
                if candidate_causal_id in consumed_causal_ids:
                    continue
                if clause_key and _normalize_clause(candidate_causal.get("clause", "")) == clause_key:
                    causal_rule_id = candidate_causal_id
                    causal = dict(candidate_causal)
                    matched_by = "normalized_clause"
                    alignment_warning = (
                        "Step 17 active rule_id was not found in Step 18M, but normalized clause matched "
                        f"Step 18M rule_id={candidate_causal_id}."
                    )
                    break
        if causal_rule_id:
            consumed_causal_ids.add(causal_rule_id)

        row = _make_reselection_row(
            rule=rule,
            causal=causal,
            eval_row=dict(eval_rows.get(rule_id, {})),
            rank=_rule_rank(rule, active_rank),
            found_in_step17=True,
            found_in_step18m=bool(causal),
            matched_by=matched_by,
            alignment_warning=alignment_warning,
            cfg=cfg,
            feedback_counts=feedback_counts,
            backup_min_trigger_count=backup_min_trigger_count,
            backfill_mixed_rules=False,
        )
        helpful_count = _safe_int(row.get("helpful_count", 0))
        harmful_count = _safe_int(row.get("harmful_count", 0))
        necessary_true_positive_count = _safe_int(row.get("necessary_true_positive_count", 0))
        grounding = _rule_grounding_features(rule, cfg)
        broad_refinement_target = _is_broad_high_coverage_rule(rule, grounding)
        pure_harmful = helpful_count == 0 and harmful_count > 0 and necessary_true_positive_count == 0
        mixed_rule = helpful_count > 0 and harmful_count > 0
        harmful_dominant_mixed = mixed_rule and harmful_count > helpful_count
        if pure_harmful:
            row["assigned_reselection_category"] = "harmful_false_positive_rule"
            row["reselection_decision"] = "remove"
            row["removal_reason"] = "pure harmful causal false-positive source"
            row["blacklist_status"] = "blacklisted_harmful"
        elif harmful_dominant_mixed:
            row["assigned_reselection_category"] = "mixed_rule"
            row["reselection_decision"] = "remove" if remove_mixed_harmful_dominant else "refine"
            row["removal_reason"] = "harmful-dominant mixed causal source" if remove_mixed_harmful_dominant else ""
            row["blacklist_status"] = "blacklisted_harmful" if remove_mixed_harmful_dominant else ""
        elif mixed_rule:
            row["assigned_reselection_category"] = "mixed_rule"
            row["reselection_decision"] = "refine"
            row["removal_reason"] = ""
            row["blacklist_status"] = ""
        elif broad_refinement_target and str(row.get("reselection_decision", "")) != "remove":
            row["assigned_reselection_category"] = (
                "weak_causal_grounding_rule"
                if bool(grounding.get("weak_causal_grounding", False))
                else str(row.get("assigned_reselection_category", "broad_high_coverage_rule"))
            )
            row["reselection_decision"] = "refine" if bool(grounding.get("weak_causal_grounding", False)) else "keep"
            row["removal_reason"] = ""
            row["blacklist_status"] = ""
        else:
            row["blacklist_status"] = ""
            row["removal_reason"] = ""
            if str(row.get("reselection_decision", "")) in {"backfill_keep", "backfill_refine_candidate"}:
                row["reselection_decision"] = "keep"
        row["selection_source"] = "original_topk_retained"
        row["refill_reason"] = ""
        row["replaced_removed_rule_id"] = ""
        if str(row.get("reselection_decision", "")) == "remove":
            removed_rows.append(row)
            if rule_id:
                blacklisted_rule_ids.add(rule_id)
            clause_key = _normalize_clause(rule.get("clause", ""))
            if clause_key:
                blacklisted_clauses.add(clause_key)
        else:
            retained_rows.append(row)
        if (
            str(row.get("assigned_reselection_category", "")) == "mixed_rule"
            or harmful_dominant_mixed
            or broad_refinement_target
            or str(row.get("assigned_reselection_category", "")) == "weak_causal_grounding_rule"
        ):
            row["suggested_action"] = (
                "refine_with_more_specific_descendant"
                if str(row.get("assigned_reselection_category", "")) in {"mixed_rule", "weak_causal_grounding_rule"}
                else "keep_but_monitor"
            )
            refinement_target_rows.append(row)
        active_rows.append(row)

    selected_rules: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    selected_rule_ids: set[str] = set()
    selected_clause_keys: set[str] = set()
    for row in retained_rows:
        rule = dict(active_rule_by_id.get(str(row.get("rule_id", "")), {}))
        if not rule:
            rule = dict(active_rule_by_clause.get(_normalize_clause(row.get("clause", "")), {}))
        if not rule:
            continue
        selected_rules.append(rule)
        selected_rows.append(row)
        selected_rule_ids.add(str(rule.get("rule_id", "")).strip())
        selected_clause_keys.add(_normalize_clause(rule.get("clause", "")))

    refilled_rows: List[Dict[str, Any]] = []
    scanned_for_refill = 0
    skipped_duplicate = 0
    skipped_blacklist = 0
    skipped_zero_positive_support = 0
    replacement_queue = [str(row.get("rule_id", "")) for row in removed_rows]

    for ranked_rule in ranked_rules:
        if len(selected_rules) >= top_k:
            break
        scanned_for_refill += 1
        rule_id = str(ranked_rule.get("rule_id", "")).strip()
        clause_key = _normalize_clause(ranked_rule.get("clause", ""))
        if rule_id in selected_rule_ids or (clause_key and clause_key in selected_clause_keys):
            skipped_duplicate += 1
            continue
        if rule_id in blacklisted_rule_ids or (clause_key and clause_key in blacklisted_clauses):
            skipped_blacklist += 1
            continue
        if skip_zero_positive_support_refill and _safe_int(ranked_rule.get("positive_support", 0)) <= 0:
            skipped_zero_positive_support += 1
            continue
        replaced_rule_id = replacement_queue.pop(0) if replacement_queue else ""
        refill_reason = "refill_removed_harmful_rule" if replaced_rule_id else "fill_to_top_k_budget"
        refill_row = {
            "rule_id": rule_id,
            "clause": str(ranked_rule.get("clause", "")),
            "original_clause": str(ranked_rule.get("clause", "")),
            "original_rank": _rule_rank(ranked_rule, len(selected_rules) + 1),
            "original_score": _rule_score(ranked_rule),
            "found_in_step18m": False,
            "missing_from_step18m": True,
            "found_in_step17": True,
            "matched_by": "ranked_pool",
            "causal_effect_status": "not_evaluated_by_18m",
            "alignment_warning": "",
            "selection_source": "ranked_pool_refill",
            "backfill_reason": "",
            "helpful_count": 0,
            "harmful_count": 0,
            "necessary_true_positive_count": 0,
            "causal_false_positive_count": 0,
            "redundant_count": 0,
            "prediction_flip_count": 0,
            "net_helpful_minus_harmful": 0,
            "dominant_influence_type": "not_evaluated_by_18m",
            "assigned_reselection_category": "ranked_pool_refill_rule",
            "reselection_decision": "refill",
            "reason": refill_reason,
            "trigger_count": 0,
            "non_decisive_contribution_count": 0,
            "weak_grounding_penalty_applied": False,
            "has_object_level_support": False,
            "uses_only_ego_motion_atoms": False,
            "uses_broad_weak_atoms": False,
            "reasoning_feedback_request_count": 0,
            "backup_explanation_candidate": False,
            "warning_message": (
                "Step 17 ranked-pool refill rule was not evaluated by Step 18M in this iteration; "
                "causal effect statistics are missing, not zero."
            ),
            "reselection_score": _rule_score(ranked_rule),
            "blacklist_status": "",
            "refill_reason": refill_reason,
            "refill_rank": len(refilled_rows) + 1,
            "replaced_removed_rule_id": replaced_rule_id,
        }
        refilled_rows.append(refill_row)
        selected_rows.append(refill_row)
        selected_rules.append(dict(ranked_rule))
        if rule_id:
            selected_rule_ids.add(rule_id)
        if clause_key:
            selected_clause_keys.add(clause_key)

    for new_rank, row in enumerate(selected_rows, start=1):
        row["refined_rank"] = new_rank

    refined_rules: List[Dict[str, Any]] = []
    for row, rule in zip(selected_rows, selected_rules):
        refined_rule = dict(rule)
        refined_rule["causal_reselection"] = {key: row.get(key, "") for key in _ROW_FIELDS}
        refined_rule["refined_rank"] = int(row.get("refined_rank", 0))
        refined_rules.append(refined_rule)

    summary_rows = active_rows + refilled_rows
    decision_counts = Counter(str(row.get("reselection_decision", "")) for row in summary_rows)
    category_counts = Counter(str(row.get("assigned_reselection_category", "")) for row in summary_rows)
    reached_top_k = len(refined_rules) == top_k
    harmful_active_rows = [row for row in active_rows if _safe_int(row.get("harmful_count", 0)) > 0]
    pure_harmful_removal_candidates = [
        row
        for row in active_rows
        if _safe_int(row.get("helpful_count", 0)) == 0
        and _safe_int(row.get("harmful_count", 0)) > 0
        and _safe_int(row.get("necessary_true_positive_count", 0)) == 0
    ]
    harmful_dominant_mixed_rows = [
        row
        for row in active_rows
        if _safe_int(row.get("helpful_count", 0)) > 0
        and _safe_int(row.get("harmful_count", 0)) > _safe_int(row.get("helpful_count", 0))
    ]
    step17_missing_from_step18m = [row for row in active_rows if not bool(row.get("found_in_step18m", False))]
    step18m_nonzero_missing_from_step17: List[Dict[str, Any]] = []
    active_clause_keys = {
        _normalize_clause(rule.get("clause", ""))
        for rule in final_rules
        if _normalize_clause(rule.get("clause", ""))
    }
    active_rule_ids = {str(rule.get("rule_id", "")).strip() for rule in final_rules if str(rule.get("rule_id", "")).strip()}
    for causal_rule_id, causal in sorted(causal_rows.items()):
        clause_key = _normalize_clause(causal.get("clause", ""))
        if causal_rule_id in consumed_causal_ids:
            continue
        if causal_rule_id in active_rule_ids or (clause_key and clause_key in active_clause_keys):
            continue
        if _has_nonzero_causal_effect(causal):
            step18m_nonzero_missing_from_step17.append(
                {
                    "rule_id": causal_rule_id,
                    "clause": str(causal.get("clause", "")),
                    "prediction_flip_count": _safe_int(causal.get("prediction_flip_count", 0)),
                    "helpful_count": _safe_int(causal.get("helpful_count", 0)),
                    "harmful_count": _safe_int(causal.get("harmful_count", 0)),
                    "necessary_true_positive_count": _safe_int(
                        causal.get("necessary_true_positive_count", 0)
                    ),
                }
            )
    warning_section = {
        "alignment_warnings": [],
        "num_step17_rules_missing_from_step18m": len(step17_missing_from_step18m),
        "step17_rules_missing_from_step18m": [_row_brief(row) for row in step17_missing_from_step18m],
        "num_step18m_nonzero_causal_effect_rules_missing_from_step17": len(
            step18m_nonzero_missing_from_step17
        ),
        "step18m_nonzero_causal_effect_rules_missing_from_step17": step18m_nonzero_missing_from_step17,
        "top_k": top_k,
        "num_initial_final_rules": len(final_rules),
        "num_removed_rules": len(removed_rows),
        "num_active_rules_with_harmful_count": len(harmful_active_rows),
        "num_pure_harmful_removal_candidates": len(pure_harmful_removal_candidates),
        "num_harmful_dominant_mixed_rules": len(harmful_dominant_mixed_rows),
        "num_refilled_rules": len(refilled_rows),
        "num_refined_final_rules": len(refined_rules),
        "num_blacklisted_rules": len(removed_rows),
        "num_refinement_targets": len(refinement_target_rows),
        "refined_final_rules_reached_top_k": reached_top_k,
        "ranked_pool_candidates_scanned_for_refill": scanned_for_refill,
        "num_refill_skipped_duplicate": skipped_duplicate,
        "num_refill_skipped_blacklist": skipped_blacklist,
        "num_refill_skipped_zero_positive_support": skipped_zero_positive_support,
    }

    _write_csv(summary_csv_path, _ROW_FIELDS, summary_rows)
    _write_csv(removed_csv_path, _REMOVED_RULE_FIELDS, removed_rows)
    _write_csv(retained_csv_path, _ROW_FIELDS, retained_rows)
    _write_csv(refilled_csv_path, _REFILLED_RULE_FIELDS, refilled_rows)
    _write_csv(refinement_targets_csv_path, _REFINEMENT_TARGET_FIELDS, refinement_target_rows)
    _write_csv(refined_csv_path, _REFINED_RULE_FIELDS, selected_rows)

    refined_result = {
        "version": _RESELECTION_VERSION,
        "config": cfg_subset,
        "selection_method": "causal_aware_top_k_rule_maintenance",
        "selection_input_stage": "step17_topk_plus_step17_ranked_pool_plus_step18m_causal_masking",
        "usage_constraints": {
            "does_not_learn_new_rules": True,
            "does_not_add_object_facts": True,
            "does_not_modify_detections": True,
            "does_not_modify_tracking": True,
            "step23_feedback_is_diagnostic_only": True,
            "validation_time_rule_selection_only": True,
        },
        "top_k": top_k,
        "num_input_rules": len(final_rules),
        "num_initial_final_rules": len(final_rules),
        "num_step17_rules": len(final_rules),
        "num_step18m_rules": len(causal_rows),
        "num_overlap_rules": sum(1 for row in active_rows if bool(row.get("found_in_step18m", False))),
        "num_final_rules": len(refined_rules),
        "num_refined_final_rules": len(refined_rules),
        "num_removed_rules": len(removed_rows),
        "num_active_rules_with_harmful_count": len(harmful_active_rows),
        "num_pure_harmful_removal_candidates": len(pure_harmful_removal_candidates),
        "num_harmful_dominant_mixed_rules": len(harmful_dominant_mixed_rows),
        "num_removed_harmful_rules": sum(
            1
            for row in removed_rows
            if str(row.get("assigned_reselection_category", "")) == "harmful_false_positive_rule"
        ),
        "num_refilled_rules": len(refilled_rows),
        "num_blacklisted_rules": len(removed_rows),
        "num_refinement_targets": len(refinement_target_rows),
        "refined_final_rules_reached_top_k": reached_top_k,
        "ranked_pool_candidates_scanned_for_refill": scanned_for_refill,
        "num_refill_skipped_duplicate": skipped_duplicate,
        "num_refill_skipped_blacklist": skipped_blacklist,
        "num_refill_skipped_zero_positive_support": skipped_zero_positive_support,
        "skip_zero_positive_support_refill": skip_zero_positive_support_refill,
        "remove_mixed_harmful_dominant": remove_mixed_harmful_dominant,
        "num_step17_rules_missing_from_step18m": len(step17_missing_from_step18m),
        "num_step18m_nonzero_causal_effect_rules_missing_from_step17": len(
            step18m_nonzero_missing_from_step17
        ),
        "num_backup_explanation_candidates": sum(
            1 for row in summary_rows if bool(row.get("backup_explanation_candidate", False))
        ),
        "num_retained_necessary_rules": sum(
            1
            for row in retained_rows
            if str(row.get("assigned_reselection_category", "")) == "necessary_positive_rule"
        ),
        "num_mixed_refinement_targets": sum(
            1
            for row in refinement_target_rows
            if str(row.get("assigned_reselection_category", "")) == "mixed_rule"
        ),
        "refilled_rules": refilled_rows,
        "removed_rules": removed_rows,
        "refinement_targets": refinement_target_rows,
        "decision_counts": dict(sorted(decision_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "warning_section": warning_section,
        "final_rules": refined_rules,
        "reselection_rows": summary_rows,
        "output_paths": {
            "result_json": str(result_path),
            "refined_final_rules_json": str(refined_json_path),
            "refined_final_rules_csv": str(refined_csv_path),
            "causal_rule_reselection_summary_csv": str(summary_csv_path),
            "removed_rules_csv": str(removed_csv_path),
            "retained_rules_csv": str(retained_csv_path),
            "refilled_rules_csv": str(refilled_csv_path),
            "refinement_targets_csv": str(refinement_targets_csv_path),
        },
    }
    with refined_json_path.open("w", encoding="utf-8") as fh:
        json.dump(refined_result, fh, indent=2)
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(refined_result, fh, indent=2)

    print(
        "  causal_rule_reselection: "
        f"input_rules={len(final_rules)} | refined_rules={len(refined_rules)} | "
        f"removed={len(removed_rows)} | refine_targets={len(refinement_target_rows)}"
    )
    refined_result["_freshly_computed"] = True
    return refined_result


def run(
    final_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    rule_level_causal_masking_results: Dict[str, Any],
    reasoning_feedback_results: Optional[Dict[str, Any]] = None,
    temporal_rule_results: Optional[Sequence[Dict[str, Any]]] = None,
    logic_atom_results: Optional[Sequence[Dict[str, Any]]] = None,
    eval_video_ids: Optional[Sequence[str]] = None,
    split_manifest: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_reselection(
        final_rule_results=final_rule_results,
        evaluation_results=evaluation_results,
        rule_level_causal_masking_results=rule_level_causal_masking_results,
        reasoning_feedback_results=reasoning_feedback_results,
        temporal_rule_results=temporal_rule_results,
        logic_atom_results=logic_atom_results,
        eval_video_ids=eval_video_ids,
        split_manifest=split_manifest,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
