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


_RESELECTION_VERSION = 3
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
    "reselection_score",
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


def _row_brief(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "rule_id": str(row.get("rule_id", "")),
        "clause": str(row.get("original_clause", row.get("clause", ""))),
    }


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
    backfill_mixed_rules = bool(cfg.get("backfill_mixed_rules", False))

    candidate_rows: List[Dict[str, Any]] = []
    rule_by_id: Dict[str, Dict[str, Any]] = {}
    rule_by_clause: Dict[str, Dict[str, Any]] = {}
    rank_by_rule_id: Dict[str, int] = {}
    consumed_causal_ids: set[str] = set()
    for rank, rule in enumerate(final_rules, start=1):
        rule_id = str(rule.get("rule_id", "")).strip()
        if rule_id:
            rule_by_id[rule_id] = rule
            rank_by_rule_id[rule_id] = rank
        clause_key = _normalize_clause(rule.get("clause", ""))
        if clause_key:
            rule_by_clause.setdefault(clause_key, rule)

    for rank, rule in enumerate(final_rules, start=1):
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id:
            continue
        matched_by = "missing_from_step18m"
        alignment_warning = ""
        causal: Dict[str, Any] = {}
        causal_rule_id = ""
        if rule_id in causal_rows:
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
                        "Step 17 rule_id was not found in Step 18M, but normalized clause matched "
                        f"Step 18M rule_id={candidate_causal_id}."
                    )
                    break
        if causal_rule_id:
            consumed_causal_ids.add(causal_rule_id)
        candidate_rows.append(
            _make_reselection_row(
                rule=rule,
                causal=causal,
                eval_row=dict(eval_rows.get(rule_id, {})),
                rank=rank,
                found_in_step17=True,
                found_in_step18m=bool(causal),
                matched_by=matched_by,
                alignment_warning=alignment_warning,
                cfg=cfg,
                feedback_counts=feedback_counts,
                backup_min_trigger_count=backup_min_trigger_count,
                backfill_mixed_rules=backfill_mixed_rules,
            )
        )

    for causal_rule_id, causal in sorted(causal_rows.items()):
        if causal_rule_id in consumed_causal_ids:
            continue
        causal_rule = _rule_from_causal_row({"rule_id": causal_rule_id, **dict(causal)})
        clause_key = _normalize_clause(causal_rule.get("clause", ""))
        matched_by = "step18m_only"
        alignment_warning = ""
        if clause_key in rule_by_clause:
            matched_by = "normalized_clause"
            alignment_warning = (
                "Step 18M rule_id was not found in Step 17 by id, but its normalized clause "
                "matches a Step 17 rule already represented in the candidate pool."
            )
        elif str(causal.get("alignment_warning", "")).strip():
            alignment_warning = str(causal.get("alignment_warning", "")).strip()
        candidate_rows.append(
            _make_reselection_row(
                rule=causal_rule,
                causal={"rule_id": causal_rule_id, **dict(causal)},
                eval_row=dict(eval_rows.get(causal_rule_id, {})),
                rank="",
                found_in_step17=False,
                found_in_step18m=True,
                matched_by=matched_by,
                alignment_warning=alignment_warning,
                cfg=cfg,
                feedback_counts=feedback_counts,
                backup_min_trigger_count=backup_min_trigger_count,
                backfill_mixed_rules=backfill_mixed_rules,
            )
        )

    summary_rows = candidate_rows
    step17_rows = [row for row in candidate_rows if bool(row.get("found_in_step17", False))]
    step18m_rows = [row for row in candidate_rows if bool(row.get("found_in_step18m", False))]
    overlap_rows = [
        row for row in candidate_rows if bool(row.get("found_in_step17", False)) and bool(row.get("found_in_step18m", False))
    ]
    missing_from_step18m_rows = [row for row in step17_rows if bool(row.get("missing_from_step18m", False))]
    step18m_missing_from_step17_rows = [row for row in step18m_rows if not bool(row.get("found_in_step17", False))]
    step18m_only_nonzero_rows = [
        row for row in step18m_missing_from_step17_rows if _safe_int(row.get("prediction_flip_count", 0)) > 0
    ]
    step18m_only_backfilled_keep_rows = [
        row for row in step18m_missing_from_step17_rows if str(row.get("reselection_decision", "")) == "backfill_keep"
    ]
    step18m_only_mixed_refinement_rows = [
        row
        for row in step18m_missing_from_step17_rows
        if str(row.get("assigned_reselection_category", "")) == "mixed_rule"
    ]
    step18m_only_harmful_not_backfilled_rows = [
        row
        for row in step18m_missing_from_step17_rows
        if str(row.get("assigned_reselection_category", "")) == "harmful_false_positive_rule"
    ]
    uncovered_nonzero_rows = [
        row
        for row in step18m_only_nonzero_rows
        if str(row.get("reselection_decision", ""))
        not in {"backfill_keep", "backfill_refine_candidate", "remove"}
    ]
    alignment_warnings: List[str] = []
    nonzero_step18m_count = sum(1 for row in step18m_rows if _safe_int(row.get("prediction_flip_count", 0)) > 0)
    if nonzero_step18m_count:
        missing_nonzero_fraction = len(step18m_only_nonzero_rows) / max(1, nonzero_step18m_count)
        if missing_nonzero_fraction > 0.10:
            alignment_warnings.append(
                "More than 10% of Step 18M nonzero-effect rules are missing from Step 17 "
                f"({len(step18m_only_nonzero_rows)}/{nonzero_step18m_count})."
            )
    if uncovered_nonzero_rows:
        alignment_warnings.append(
            "Some Step 18M-only rules with prediction_flip_count > 0 were neither backfilled nor marked "
            f"for refinement: {[str(row.get('rule_id', '')) for row in uncovered_nonzero_rows]}"
        )
    warning_section = {
        "num_step17_rules": len(final_rules),
        "num_step18m_rules": len(causal_rows),
        "num_overlap_rules": len(overlap_rows),
        "num_step17_missing_from_18m": len(missing_from_step18m_rows),
        "num_step18m_missing_from_17": len(step18m_missing_from_step17_rows),
        "num_step18m_only_nonzero_effect_rules": len(step18m_only_nonzero_rows),
        "num_step18m_only_backfilled_keep_rules": len(step18m_only_backfilled_keep_rows),
        "num_step18m_only_mixed_refinement_targets": len(step18m_only_mixed_refinement_rows),
        "num_step18m_only_harmful_not_backfilled": len(step18m_only_harmful_not_backfilled_rows),
        "backfill_mixed_rules_enabled": backfill_mixed_rules,
        "num_step17_rules_missing_from_step18m": len(missing_from_step18m_rows),
        "step17_rules_missing_from_step18m": [_row_brief(row) for row in missing_from_step18m_rows],
        "num_step18m_nonzero_causal_effect_rules_missing_from_step17": len(step18m_only_nonzero_rows),
        "step18m_nonzero_causal_effect_rules_missing_from_step17": [
            _row_brief(row) for row in step18m_only_nonzero_rows
        ],
        "alignment_warnings": alignment_warnings,
    }

    selectable_rows = [
        row
        for row in candidate_rows
        if str(row.get("reselection_decision", "")) != "remove"
        and (
            bool(row.get("found_in_step17", False))
            or str(row.get("reselection_decision", "")) == "backfill_keep"
            or (backfill_mixed_rules and str(row.get("reselection_decision", "")) == "backfill_refine_candidate")
        )
    ]
    def _selection_priority(item: Dict[str, Any]) -> int:
        decision = str(item.get("reselection_decision", ""))
        category = str(item.get("assigned_reselection_category", ""))
        if decision == "backfill_keep":
            return 0
        if bool(item.get("found_in_step17", False)) and category == "necessary_positive_rule":
            return 1
        if bool(item.get("found_in_step17", False)) and decision in {"keep", "refine"}:
            return 2
        if decision == "backfill_refine_candidate":
            return 3
        if decision in {"keep_for_review", "backup_explanation"}:
            return 4
        return 5

    def _selection_key(item: Dict[str, Any]) -> Tuple[int, float, int, str]:
        return (
            _selection_priority(item),
            -_safe_float(item.get("reselection_score", 0.0)),
            _safe_int(item.get("original_rank", 10**9), 10**9),
            str(item.get("rule_id", "")),
        )

    selected_rows = []
    selected_seen: set[str] = set()
    for row in sorted(selectable_rows, key=_selection_key):
        if len(selected_rows) >= top_k:
            break
        rule_id = str(row.get("rule_id", ""))
        if rule_id not in selected_seen:
            selected_rows.append(row)
            selected_seen.add(rule_id)
    selected_rule_ids = {str(row.get("rule_id", "")) for row in selected_rows}
    for new_rank, row in enumerate(selected_rows, start=1):
        row["refined_rank"] = new_rank

    refined_rules: List[Dict[str, Any]] = []
    for row in selected_rows:
        rule = dict(rule_by_id.get(str(row.get("rule_id", "")), {}))
        if not rule:
            rule = _rule_from_causal_row(
                {
                    "rule_id": row.get("rule_id", ""),
                    "clause": row.get("clause", row.get("original_clause", "")),
                    "confidence": row.get("original_score", 0.0),
                }
            )
        rule["causal_reselection"] = {key: row.get(key, "") for key in _ROW_FIELDS}
        rule["refined_rank"] = int(row.get("refined_rank", 0))
        refined_rules.append(rule)

    retained_rows = [row for row in candidate_rows if str(row.get("rule_id", "")) in selected_rule_ids]
    removed_rows = [
        row
        for row in candidate_rows
        if str(row.get("reselection_decision", "")) == "remove"
    ]
    refinement_target_rows = [
        row
        for row in candidate_rows
        if str(row.get("reselection_decision", "")) in {"refine", "backfill_refine_candidate"}
        or (
            _is_refinement_target_candidate(
                {
                    "uses_broad_weak_atoms": row.get("uses_broad_weak_atoms", False),
                    "uses_only_ego_motion_atoms": row.get("uses_only_ego_motion_atoms", False),
                }
            )
            and str(row.get("reselection_decision", "")) != "remove"
        )
    ]
    decision_counts = Counter(str(row.get("reselection_decision", "")) for row in candidate_rows)
    category_counts = Counter(str(row.get("assigned_reselection_category", "")) for row in candidate_rows)

    _write_csv(summary_csv_path, _ROW_FIELDS, summary_rows)
    _write_csv(removed_csv_path, _ROW_FIELDS, removed_rows)
    _write_csv(retained_csv_path, _ROW_FIELDS, retained_rows)
    _write_csv(refinement_targets_csv_path, _ROW_FIELDS, refinement_target_rows)
    _write_csv(refined_csv_path, _ROW_FIELDS + ["refined_rank"], selected_rows)

    refined_result = {
        "version": _RESELECTION_VERSION,
        "config": cfg_subset,
        "selection_method": "causal_masking_guided_reselection",
        "selection_input_stage": "step17_final_rules_plus_step18m_causal_masking",
        "usage_constraints": {
            "does_not_learn_new_rules": True,
            "does_not_add_object_facts": True,
            "does_not_modify_detections": True,
            "does_not_modify_tracking": True,
            "step23_feedback_is_diagnostic_only": True,
            "validation_time_rule_selection_only": True,
        },
        "num_input_rules": len(final_rules),
        "num_step17_rules": len(final_rules),
        "num_step18m_rules": len(causal_rows),
        "num_overlap_rules": len(overlap_rows),
        "num_step17_missing_from_18m": len(missing_from_step18m_rows),
        "num_step18m_missing_from_17": len(step18m_missing_from_step17_rows),
        "num_step18m_only_nonzero_effect_rules": len(step18m_only_nonzero_rows),
        "num_step18m_only_backfilled_keep_rules": len(step18m_only_backfilled_keep_rows),
        "num_step18m_only_mixed_refinement_targets": len(step18m_only_mixed_refinement_rows),
        "num_step18m_only_harmful_not_backfilled": len(step18m_only_harmful_not_backfilled_rows),
        "backfill_mixed_rules_enabled": backfill_mixed_rules,
        "num_final_rules": len(refined_rules),
        "num_removed_rules": len(removed_rows),
        "num_removed_harmful_rules": sum(
            1
            for row in removed_rows
            if str(row.get("assigned_reselection_category", "")) == "harmful_false_positive_rule"
        ),
        "num_step17_rules_missing_from_step18m": len(missing_from_step18m_rows),
        "num_step18m_nonzero_causal_effect_rules_missing_from_step17": len(step18m_only_nonzero_rows),
        "num_backup_explanation_candidates": sum(
            1 for row in candidate_rows if bool(row.get("backup_explanation_candidate", False))
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
