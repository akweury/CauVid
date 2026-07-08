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


_RESELECTION_VERSION = 2
_ROW_FIELDS = [
    "rule_id",
    "original_clause",
    "original_rank",
    "original_score",
    "found_in_step18m",
    "missing_from_step18m",
    "found_in_step17",
    "causal_effect_status",
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


def _body_atoms(rule: Dict[str, Any]) -> List[str]:
    for key in ("body_atom_templates", "body_atoms", "antecedent_atoms"):
        value = rule.get(key)
        if isinstance(value, list):
            atoms = [str(atom).strip() for atom in value if str(atom).strip()]
            if atoms:
                return atoms
    clause = str(rule.get("clause", ""))
    if ":-" not in clause:
        return []
    body = clause.split(":-", 1)[1]
    body = body.rsplit(".", 1)[0]
    return [part.strip() for part in body.split(",") if part.strip()]


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
    return {str(row.get("rule_id", "")): dict(row) for row in rows if str(row.get("rule_id", ""))}


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
        if harmful_count == 0:
            return "keep", "masking shows necessary true-positive support without harmful flips", False
        return "keep", "rule preserves positive coverage despite some harmful contribution", False
    if category == "mixed_rule":
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
    return score


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

    step17_summary_rows: List[Dict[str, Any]] = []
    rule_by_id: Dict[str, Dict[str, Any]] = {}
    for rank, rule in enumerate(final_rules, start=1):
        rule_id = str(rule.get("rule_id", "")).strip()
        if not rule_id:
            continue
        rule_by_id[rule_id] = rule
        found_in_step18m = rule_id in causal_rows
        causal = dict(causal_rows.get(rule_id, {}))
        eval_row = dict(eval_rows.get(rule_id, {}))
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
        )
        if category == "weak_causal_grounding_rule" and feedback_counts.get(rule_id, 0) > 0 and decision != "remove":
            decision = "refine"
            reason = f"{reason}; Step 23 feedback also flagged related examples"
        warning_message = ""
        if not found_in_step18m:
            warning_message = (
                "Step 17 rule_id was not found in Step 18M causal masking summary; "
                "causal effect statistics are missing, not zero."
            )
        row = {
            "rule_id": rule_id,
            "original_clause": str(rule.get("clause", causal.get("clause", ""))),
            "original_rank": rank,
            "original_score": _safe_float(rule.get("score", rule.get("confidence", causal.get("confidence", 0.0)))),
            "found_in_step18m": bool(found_in_step18m),
            "missing_from_step18m": bool(not found_in_step18m),
            "found_in_step17": True,
            "causal_effect_status": causal_effect_status,
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
        step17_summary_rows.append(row)

    step17_rule_ids = set(rule_by_id)
    step18m_nonzero_missing_from_step17_rows: List[Dict[str, Any]] = []
    for causal_rule_id, causal in sorted(causal_rows.items()):
        if causal_rule_id in step17_rule_ids or not _has_nonzero_causal_effect(causal):
            continue
        step18m_nonzero_missing_from_step17_rows.append(
            {
                "rule_id": causal_rule_id,
                "original_clause": str(causal.get("clause", "")),
                "original_rank": "",
                "original_score": _safe_float(causal.get("confidence", 0.0)),
                "found_in_step18m": True,
                "missing_from_step18m": False,
                "found_in_step17": False,
                "causal_effect_status": "nonzero_causal_effect_without_step17_rule",
                "helpful_count": _safe_int(causal.get("helpful_count", 0)),
                "harmful_count": _safe_int(causal.get("harmful_count", 0)),
                "necessary_true_positive_count": _safe_int(causal.get("necessary_true_positive_count", 0)),
                "causal_false_positive_count": _safe_int(causal.get("causal_false_positive_count", 0)),
                "redundant_count": _safe_int(causal.get("redundant_count", 0)),
                "prediction_flip_count": _safe_int(causal.get("prediction_flip_count", 0)),
                "net_helpful_minus_harmful": _safe_int(causal.get("helpful_count", 0))
                - _safe_int(causal.get("harmful_count", 0)),
                "dominant_influence_type": str(causal.get("dominant_influence_type", "none")),
                "assigned_reselection_category": "causal_effect_without_step17_rule",
                "reselection_decision": "warning_only",
                "reason": "Step 18M reports nonzero causal effect for a rule_id absent from Step 17 final rules",
                "trigger_count": _safe_int(causal.get("trigger_count", 0)),
                "non_decisive_contribution_count": _safe_int(causal.get("non_decisive_contribution_count", 0)),
                "weak_grounding_penalty_applied": False,
                "has_object_level_support": False,
                "uses_only_ego_motion_atoms": False,
                "uses_broad_weak_atoms": False,
                "reasoning_feedback_request_count": int(feedback_counts.get(causal_rule_id, 0)),
                "backup_explanation_candidate": False,
                "warning_message": "Step 18M rule with nonzero causal effect was not found in Step 17 final rules.",
                "reselection_score": "",
            }
        )
    summary_rows = step17_summary_rows + step18m_nonzero_missing_from_step17_rows
    missing_from_step18m_rows = [row for row in step17_summary_rows if bool(row.get("missing_from_step18m", False))]
    warning_section = {
        "num_step17_rules_missing_from_step18m": len(missing_from_step18m_rows),
        "step17_rules_missing_from_step18m": [_row_brief(row) for row in missing_from_step18m_rows],
        "num_step18m_nonzero_causal_effect_rules_missing_from_step17": len(
            step18m_nonzero_missing_from_step17_rows
        ),
        "step18m_nonzero_causal_effect_rules_missing_from_step17": [
            _row_brief(row) for row in step18m_nonzero_missing_from_step17_rows
        ],
    }

    selected_rows = [
        row
        for row in sorted(
            step17_summary_rows,
            key=lambda item: (
                str(item.get("reselection_decision", "")) == "remove",
                -_safe_float(item.get("reselection_score", 0.0)),
                _safe_int(item.get("original_rank", 10**9), 10**9),
            ),
        )
        if str(row.get("reselection_decision", "")) != "remove"
    ][:top_k]
    selected_rule_ids = {str(row.get("rule_id", "")) for row in selected_rows}
    for new_rank, row in enumerate(selected_rows, start=1):
        row["refined_rank"] = new_rank

    refined_rules: List[Dict[str, Any]] = []
    for row in selected_rows:
        rule = dict(rule_by_id.get(str(row.get("rule_id", "")), {}))
        rule["causal_reselection"] = {key: row.get(key, "") for key in _ROW_FIELDS}
        rule["refined_rank"] = int(row.get("refined_rank", 0))
        refined_rules.append(rule)

    retained_rows = [row for row in step17_summary_rows if str(row.get("rule_id", "")) in selected_rule_ids]
    removed_rows = [row for row in step17_summary_rows if str(row.get("reselection_decision", "")) == "remove"]
    refinement_target_rows = [
        row
        for row in step17_summary_rows
        if str(row.get("reselection_decision", "")) == "refine"
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
    decision_counts = Counter(str(row.get("reselection_decision", "")) for row in step17_summary_rows)
    category_counts = Counter(str(row.get("assigned_reselection_category", "")) for row in step17_summary_rows)

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
        "num_final_rules": len(refined_rules),
        "num_removed_rules": len(removed_rows),
        "num_removed_harmful_rules": sum(
            1
            for row in removed_rows
            if str(row.get("assigned_reselection_category", "")) == "harmful_false_positive_rule"
        ),
        "num_step17_rules_missing_from_step18m": len(missing_from_step18m_rows),
        "num_step18m_nonzero_causal_effect_rules_missing_from_step17": len(
            step18m_nonzero_missing_from_step17_rows
        ),
        "num_backup_explanation_candidates": sum(
            1 for row in step17_summary_rows if bool(row.get("backup_explanation_candidate", False))
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
