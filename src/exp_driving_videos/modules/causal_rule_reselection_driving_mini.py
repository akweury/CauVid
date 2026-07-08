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


_RESELECTION_VERSION = 4
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


def process_reselection(
    final_rule_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    rule_level_causal_masking_results: Dict[str, Any],
    reasoning_feedback_results: Optional[Dict[str, Any]] = None,
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
    cfg: Optional[Dict[str, Any]] = None,
    output_root: Optional[Path] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    return process_reselection(
        final_rule_results=final_rule_results,
        evaluation_results=evaluation_results,
        rule_level_causal_masking_results=rule_level_causal_masking_results,
        reasoning_feedback_results=reasoning_feedback_results,
        cfg=cfg,
        output_root=output_root,
        force_recompute=force_recompute,
    )
