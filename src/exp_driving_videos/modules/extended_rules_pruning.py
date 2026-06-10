"""
Pruning strategies for extended temporal rules.

This module is intentionally small but structured as a strategy dispatcher so
additional pruning rules can be added without changing the step-16 loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _normalize_strategy_name(strategy: Optional[str]) -> str:
    text = str(strategy or "").strip().lower()
    return text or "none"


def normalize_strategy_names(strategies: Optional[Sequence[str]]) -> List[str]:
    normalized: List[str] = []
    for strategy in list(strategies or []):
        name = _normalize_strategy_name(strategy)
        if not name or name in normalized:
            continue
        normalized.append(name)
    return normalized


def _prune_none(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "kept": True,
        "prune_status": "disabled",
        "prune_reason": "",
    }


def _prune_empty_evidence(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    total_firings = int(metrics.get("total_firings", 0))
    positive_firings = int(metrics.get("positive_firings", 0))

    if total_firings <= 0:
        return {
            "kept": False,
            "prune_status": "pruned",
            "prune_reason": "empty_evidence",
        }
    if positive_firings <= 0:
        return {
            "kept": False,
            "prune_status": "pruned",
            "prune_reason": "no_positive_firings",
        }
    return {
        "kept": True,
        "prune_status": "kept",
        "prune_reason": "",
    }


def _prune_same_firings_as_parent(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    positive_firings = int(metrics.get("positive_firings", 0))
    negative_firings = int(metrics.get("negative_firings", 0))
    parent_positive_firings = int(metrics.get("parent_positive_firings", -1))
    parent_negative_firings = int(metrics.get("parent_negative_firings", -1))

    if (
        positive_firings == parent_positive_firings
        and negative_firings == parent_negative_firings
    ):
        return {
            "kept": False,
            "prune_status": "pruned",
            "prune_reason": "same_firings_as_parent",
        }
    return {
        "kept": True,
        "prune_status": "kept",
        "prune_reason": "",
    }


def _prune_same_confidence_smaller_evidence(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(metrics.get("is_last_round", False)):
        return {
            "kept": True,
            "prune_status": "deferred",
            "prune_reason": "",
        }

    confidence = float(metrics.get("confidence", 0.0))
    parent_confidence = float(metrics.get("parent_confidence", float("nan")))
    total_firings = int(metrics.get("total_firings", 0))
    parent_total_firings = int(metrics.get("parent_total_firings", -1))

    if confidence == parent_confidence and 0 <= total_firings < parent_total_firings:
        return {
            "kept": False,
            "prune_status": "pruned",
            "prune_reason": "same_confidence_smaller_evidence",
        }
    return {
        "kept": True,
        "prune_status": "kept",
        "prune_reason": "",
    }


def _prune_low_evidence(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    positive_support = int(metrics.get("positive_support", 0))
    negative_support = int(metrics.get("negative_support", 0))
    min_positive_support_to_extend = int(metrics.get("min_positive_support_to_extend", 1))

    if positive_support < min_positive_support_to_extend and negative_support == 0:
        return {
            "kept": False,
            "prune_status": "pruned",
            "prune_reason": "low_evidence",
        }
    return {
        "kept": True,
        "prune_status": "kept",
        "prune_reason": "",
    }


def apply_pruning_strategy(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
    strategy: Optional[str],
) -> Dict[str, Any]:
    strategy_name = _normalize_strategy_name(strategy)
    if strategy_name in {"none", "disabled", "off", "false"}:
        decision = _prune_none(rule, metrics)
    elif strategy_name == "low_evidence":
        decision = _prune_low_evidence(rule, metrics)
    elif strategy_name == "empty_evidence":
        decision = _prune_empty_evidence(rule, metrics)
    elif strategy_name == "same_firings_as_parent":
        decision = _prune_same_firings_as_parent(rule, metrics)
    elif strategy_name == "same_confidence_smaller_evidence":
        if not bool(metrics.get("same_confidence_smaller_evidence_enabled", True)):
            decision = _prune_none(rule, metrics)
        else:
            decision = _prune_same_confidence_smaller_evidence(rule, metrics)
    else:
        raise ValueError(f"Unsupported extended-rule pruning strategy: {strategy}")

    out = dict(decision)
    out["strategy"] = strategy_name
    return out


def apply_pruning_strategies(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
    strategies: Optional[Sequence[str]],
) -> Dict[str, Any]:
    strategy_names = normalize_strategy_names(strategies)
    if not strategy_names:
        return apply_pruning_strategy(rule, metrics, "none")

    decisions: List[Dict[str, Any]] = []
    for strategy_name in strategy_names:
        decision = apply_pruning_strategy(rule, metrics, strategy_name)
        decisions.append(decision)
        if not bool(decision.get("kept", False)):
            out = dict(decision)
            out["evaluated_strategies"] = strategy_names
            out["decisions"] = decisions
            return out

    out = dict(decisions[-1])
    out["evaluated_strategies"] = strategy_names
    out["decisions"] = decisions
    return out


def apply_parent_pruning_strategies(
    rule: Dict[str, Any],
    metrics: Dict[str, Any],
    strategies: Optional[Sequence[str]],
) -> Dict[str, Any]:
    strategy_names = normalize_strategy_names(strategies)
    if not strategy_names:
        return apply_pruning_strategy(rule, metrics, "none")

    decisions: List[Dict[str, Any]] = []
    for strategy_name in strategy_names:
        decision = _prune_none(rule, metrics)
        decision["strategy"] = strategy_name
        decisions.append(decision)
        if not bool(decision.get("kept", False)):
            out = dict(decision)
            out["evaluated_strategies"] = strategy_names
            out["decisions"] = decisions
            return out

    out = dict(decisions[-1])
    out["evaluated_strategies"] = strategy_names
    out["decisions"] = decisions
    return out
