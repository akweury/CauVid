"""Optional, fail-open Weights & Biases tracking for the exp_july pipeline."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_AUDIT_ROOT_PATTERNS = (
    "*manifest*.json",
    "*conflict*.json",
    "candidate*_policy*.json",
    "promotion_decision*.json",
)
_AUDIT_SUBDIR_PATTERNS = (
    ("epoch_reviews", "*.json"),
    ("policies", "*.json"),
    ("statistics", "candidate_table*.json"),
)
_WANDB_PUBLIC_CLOUD_API = "https://api.wandb.ai"
_WANDB_PUBLIC_CLOUD_SETTINGS = "https://wandb.ai/settings"


def _truthy(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int = 0,
    fallback_names=(),
) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        raw_value = next(
            (
                os.environ[fallback_name]
                for fallback_name in fallback_names
                if fallback_name in os.environ
            ),
            str(default),
        )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value == value and abs(value) != float("inf") else default


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def _json_fingerprint(value: Any) -> Optional[str]:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(payload).hexdigest()


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if result == result and abs(result) != float("inf") else None
    return None


def _flatten_numeric(
    value: Any,
    prefix: str,
    *,
    depth: int = 0,
    max_depth: int = 4,
    limit: int = 200,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if len(result) >= limit:
        return result
    number = _numeric(value)
    if number is not None:
        return {prefix: number}
    if depth >= max_depth:
        return result
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            if len(result) >= limit:
                break
            child = _flatten_numeric(
                value[key],
                f"{prefix}/{_slug(key)}",
                depth=depth + 1,
                max_depth=max_depth,
                limit=limit - len(result),
            )
            result.update(child)
    return result


class WandbTracker:
    """Small adapter that never lets observability failures stop the pipeline."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        enabled: Optional[bool] = None,
        project: Optional[str] = None,
        run_name: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        self.enabled = (
            _truthy(os.environ.get("CAUVID_WANDB_ENABLED"), True)
            if enabled is None
            else bool(enabled)
        )
        self.project = (
            project
            or os.environ.get("CAUVID_WANDB_PROJECT")
            or os.environ.get("WANDB_PROJECT")
            or "cauvid-exp-july"
        )
        self.run_name = (
            run_name
            or os.environ.get("CAUVID_WANDB_RUN_NAME")
            or None
        )
        self.mode = (
            mode
            or os.environ.get("CAUVID_WANDB_MODE")
            or os.environ.get("WANDB_MODE")
            or "online"
        )
        self.base_url = (
            os.environ.get("CAUVID_WANDB_BASE_URL")
            or _WANDB_PUBLIC_CLOUD_API
        ).rstrip("/")
        # Do not inherit a stale WANDB_BASE_URL pointing at wandb/local.
        # CauVid uses hosted W&B unless its dedicated override is explicit.
        os.environ["WANDB_BASE_URL"] = self.base_url
        self.entity = (
            os.environ.get("CAUVID_WANDB_ENTITY")
            or os.environ.get("WANDB_ENTITY")
            or None
        )
        self.group = os.environ.get("CAUVID_WANDB_GROUP") or None
        self.tags = [
            value.strip()
            for value in os.environ.get("CAUVID_WANDB_TAGS", "").split(",")
            if value.strip()
        ]
        pipeline_root = Path(
            os.environ.get("CAUVID_PIPELINE_OUTPUT_PATH", Path.cwd())
        ).expanduser().absolute()
        self.output_root = pipeline_root
        wandb_dir = Path(
            os.environ.get("CAUVID_WANDB_DIR")
            or os.environ.get("WANDB_DIR")
            or "wandb"
        ).expanduser()
        if not wandb_dir.is_absolute():
            wandb_dir = pipeline_root / wandb_dir
        self.wandb_dir = wandb_dir.absolute()
        self.max_videos = _env_int("CAUVID_WANDB_MAX_VIDEOS", 4)
        self.max_artifact_files = _env_int(
            "CAUVID_WANDB_MAX_ARTIFACT_FILES", 200
        )
        self.init_timeout_seconds = _env_int(
            "CAUVID_WANDB_INIT_TIMEOUT_SECONDS",
            30,
            minimum=1,
            fallback_names=("WANDB_INIT_TIMEOUT",),
        )
        self.module = None
        self.run = None
        self.step_index = 0
        self.successful_steps = 0
        self.started_at = time.perf_counter()
        self.errors = []
        self._logged_media_paths = set()
        self._logged_media_video_ids = set()
        self._manifest_paths = set()
        self._discovered_manifest_paths = set()
        self._seen_manifest_payloads = set()
        self._seen_manifest_object_ids = set()
        if not self.enabled:
            return
        if (
            self.mode == "online"
            and self.base_url == _WANDB_PUBLIC_CLOUD_API
            and not os.environ.get("WANDB_API_KEY")
        ):
            print(
                "[wandb] Hosted cloud login required. Sign in at "
                f"{_WANDB_PUBLIC_CLOUD_SETTINGS}, create/copy an API key, "
                "then export WANDB_API_KEY on this server.",
                file=sys.stderr,
                flush=True,
            )
        try:
            self.module = importlib.import_module("wandb")
            self.wandb_dir.mkdir(parents=True, exist_ok=True)
            init_kwargs = {
                "project": self.project,
                "name": self.run_name,
                "entity": self.entity,
                "group": self.group,
                "tags": self.tags or None,
                "mode": self.mode,
                "dir": str(self.wandb_dir),
                "job_type": "pipeline",
                "config": config,
            }
            settings_factory = getattr(self.module, "Settings", None)
            if callable(settings_factory):
                init_kwargs["settings"] = settings_factory(
                    init_timeout=self.init_timeout_seconds
                )
            self.run = self.module.init(
                **{key: value for key, value in init_kwargs.items() if value is not None}
            )
            if self.run is not None and hasattr(self.run, "summary"):
                self.run.summary["pipeline/status"] = "running"
                self.run.summary["pipeline/tracking_mode"] = self.mode
                self.run.summary["pipeline/wandb_base_url"] = self.base_url
                run_url = getattr(self.run, "url", None)
                if isinstance(run_url, str) and run_url.startswith("http"):
                    self.run.summary["pipeline/wandb_web_url"] = run_url
                    print(
                        f"[wandb] Web dashboard: {run_url}",
                        flush=True,
                    )
        except Exception as exc:
            self._record_error("init", exc)
            self.module = None
            self.run = None

    @property
    def active(self) -> bool:
        return self.run is not None

    def _record_error(self, operation: str, error: Exception) -> None:
        message = f"{operation}: {type(error).__name__}: {error}"
        self.errors.append(message)
        print(f"[wandb][warning] {message}", file=sys.stderr, flush=True)

    def log_state(
        self,
        step_name: str,
        state: Dict[str, Any],
        *,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Log compact numeric state and capped Step 8H videos, then return state."""
        if not self.active:
            return state
        self.step_index += 1
        self.successful_steps += 1
        prefix = f"steps/{_slug(step_name)}"
        video_count = len(state.get("videos") or [])
        metrics: Dict[str, Any] = {
            "pipeline/step_index": self.step_index,
            "pipeline/step_name": step_name,
            "pipeline/stage_status": "completed",
            "pipeline/stage_video_count": video_count,
            f"{prefix}/video_count": video_count,
        }
        if duration_seconds is not None:
            metrics["pipeline/stage_duration_seconds"] = float(duration_seconds)
            metrics[f"{prefix}/duration_seconds"] = float(duration_seconds)
        for key, value in state.items():
            if key == "videos":
                continue
            number = _numeric(value)
            if number is not None:
                metrics[f"{prefix}/{_slug(key)}"] = number
            if isinstance(value, (list, tuple, set)):
                metrics[f"{prefix}/{_slug(key)}_count"] = len(value)
        new_manifest_paths = []
        try:
            new_manifest_paths = self._capture_output_paths(step_name, state)
        except Exception as exc:
            self._record_error(f"capture_outputs:{step_name}", exc)
        try:
            self._append_manifest_metrics(
                metrics,
                prefix,
                state,
                new_manifest_paths,
            )
        except Exception as exc:
            self._record_error(f"manifest_metrics:{step_name}", exc)
        try:
            self.run.log(metrics, step=self.step_index)
            if step_name.lower().lstrip("0").startswith("8h"):
                self._log_step8h_media(state)
        except Exception as exc:
            self._record_error(f"log_state:{step_name}", exc)
        return state

    def log_failure(
        self,
        step_name: str,
        error: BaseException,
        *,
        duration_seconds: Optional[float] = None,
    ) -> None:
        """Record a failed stage without changing the exception seen by callers."""
        if not self.active:
            return
        self.step_index += 1
        prefix = f"steps/{_slug(step_name)}"
        metrics: Dict[str, Any] = {
            "pipeline/step_index": self.step_index,
            "pipeline/step_name": step_name,
            "pipeline/stage_status": "failed",
            "pipeline/failed_step": step_name,
            "pipeline/error_type": type(error).__name__,
            "pipeline/error": str(error)[:1000],
            f"{prefix}/failed": 1,
        }
        if duration_seconds is not None:
            metrics["pipeline/stage_duration_seconds"] = float(duration_seconds)
            metrics[f"{prefix}/duration_seconds"] = float(duration_seconds)
        try:
            self.run.log(metrics, step=self.step_index)
            if hasattr(self.run, "summary"):
                self.run.summary["pipeline/failed_step"] = step_name
                self.run.summary["pipeline/failed_step_duration_seconds"] = (
                    float(duration_seconds)
                    if duration_seconds is not None
                    else None
                )
        except Exception as exc:
            self._record_error(f"log_failure:{step_name}", exc)

    def _append_manifest_metrics(
        self,
        metrics: Dict[str, Any],
        prefix: str,
        state: Dict[str, Any],
        manifest_paths,
    ) -> None:
        for path in manifest_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self._record_error(f"read_manifest:{path}", exc)
                continue
            if not isinstance(payload, dict):
                continue
            fingerprint = _json_fingerprint(payload)
            if fingerprint:
                self._seen_manifest_payloads.add(fingerprint)
            metrics.update(
                _flatten_numeric(
                    payload,
                    f"{prefix}/manifests/{_slug(path.stem)}",
                    limit=max(0, 250 - len(metrics)),
                )
            )
        for key, value in state.items():
            if not isinstance(value, dict) or not key.endswith("_manifest"):
                continue
            object_id = id(value)
            if object_id in self._seen_manifest_object_ids:
                continue
            self._seen_manifest_object_ids.add(object_id)
            fingerprint = _json_fingerprint(value)
            if fingerprint and fingerprint in self._seen_manifest_payloads:
                continue
            if fingerprint:
                self._seen_manifest_payloads.add(fingerprint)
            metrics.update(
                _flatten_numeric(
                    value,
                    f"{prefix}/manifests/{_slug(key)}",
                    limit=max(0, 250 - len(metrics)),
                )
            )

    def _capture_output_paths(
        self,
        step_name: str,
        state: Dict[str, Any],
    ):
        summary_count = 0
        new_manifest_paths = []
        for key, value in state.items():
            if (
                isinstance(value, (str, Path))
                and value
                and (key.endswith("_output_root") or key.endswith("_path"))
            ):
                if hasattr(self.run, "summary") and summary_count < 30:
                    self.run.summary[
                        f"outputs/{_slug(step_name)}/{_slug(key)}"
                    ] = str(value)
                    summary_count += 1
                path = Path(value)
                if key.endswith("_output_root") and path.is_dir():
                    manifests = sorted(path.glob("*manifest*.json"))
                    for manifest_path in manifests:
                        if manifest_path not in self._discovered_manifest_paths:
                            self._discovered_manifest_paths.add(manifest_path)
                            new_manifest_paths.append(manifest_path)
                    self._collect_audit_paths(path)
                elif (
                    path.is_file()
                    and "manifest" in path.name
                    and path.suffix.lower() == ".json"
                ):
                    self._manifest_paths.add(path)
                    if path not in self._discovered_manifest_paths:
                        self._discovered_manifest_paths.add(path)
                        new_manifest_paths.append(path)
        return new_manifest_paths

    def _collect_audit_paths(self, root: Path) -> None:
        if self.max_artifact_files <= len(self._manifest_paths):
            return
        candidates = set()
        for pattern in _AUDIT_ROOT_PATTERNS:
            candidates.update(root.glob(pattern))
        for subdir, pattern in _AUDIT_SUBDIR_PATTERNS:
            candidates.update((root / subdir).glob(pattern))
        if root.name in {"epoch_reviews", "policies", "statistics"}:
            candidates.update(root.glob("*.json"))
        for path in sorted(candidates):
            if not path.is_file():
                continue
            self._manifest_paths.add(path)
            if len(self._manifest_paths) >= self.max_artifact_files:
                break

    def _log_step8h_media(self, state: Dict[str, Any]) -> None:
        if self.max_videos <= 0 or self.module is None:
            return
        media = {}
        for row in state.get("relative_motion_visualizations", []):
            if not isinstance(row, dict):
                continue
            video_id = str(row.get("video_id", "")) or "unknown"
            if video_id in self._logged_media_video_ids:
                continue
            path = str(row.get("visualization_path", ""))
            if (
                not path
                or path in self._logged_media_paths
                or not Path(path).exists()
            ):
                continue
            key = (
                f"media/step8h/{_slug(video_id)}"
                f"_track_{_slug(row.get('track_id', 'unknown'))}"
            )
            media[key] = self.module.Video(path, format="mp4")
            self._logged_media_paths.add(path)
            self._logged_media_video_ids.add(video_id)
            if len(media) >= self.max_videos:
                break
        if media:
            self.run.log(media, step=self.step_index)

    def _log_audit_artifact(self) -> None:
        if (
            not self.active
            or self.module is None
            or self.max_artifact_files <= 0
        ):
            return
        paths = sorted(path for path in self._manifest_paths if path.exists())[
            : self.max_artifact_files
        ]
        if not paths:
            return
        run_id = _slug(getattr(self.run, "id", "local"))
        artifact = self.module.Artifact(
            f"cauvid-exp-july-{run_id}-audit",
            type="pipeline-audit",
            metadata={"file_count": len(paths)},
        )
        for path in paths:
            try:
                artifact_name = str(path.relative_to(self.output_root))
            except ValueError:
                artifact_name = f"external/{_slug(path.parent)}/{path.name}"
            artifact.add_file(
                str(path),
                name=artifact_name,
            )
        self.run.log_artifact(artifact)

    def finish(self, *, status: str, error: Optional[BaseException] = None) -> None:
        if not self.active:
            return
        exit_code = 0 if status == "completed" else 1
        final = {
            "pipeline/step_index": self.step_index,
            "pipeline/status": status,
            "pipeline/duration_seconds": time.perf_counter() - self.started_at,
        }
        if error is not None:
            final["pipeline/error_type"] = type(error).__name__
            final["pipeline/error"] = str(error)[:1000]
        try:
            self.run.log(final, step=self.step_index + 1)
        except Exception as exc:
            self._record_error("finish_log", exc)
        try:
            if hasattr(self.run, "summary"):
                self.run.summary["pipeline/status"] = status
                self.run.summary["pipeline/steps_logged"] = self.successful_steps
                self.run.summary["pipeline/stages_attempted"] = self.step_index
                self.run.summary["pipeline/tracking_errors"] = list(self.errors)
        except Exception as exc:
            self._record_error("finish_summary", exc)
        try:
            self._log_audit_artifact()
        except Exception as exc:
            self._record_error("finish_artifact", exc)
        try:
            self.run.finish(exit_code=exit_code)
        except TypeError:
            try:
                self.run.finish()
            except Exception as exc:
                self._record_error("finish", exc)
        except Exception as exc:
            self._record_error("finish", exc)
        finally:
            self.run = None


def create_tracker(
    *,
    video_ids: Optional[Iterable[str]],
    video_count: Optional[int],
    rounds: int,
    max_step: int,
    enabled: Optional[bool] = None,
    project: Optional[str] = None,
    run_name: Optional[str] = None,
    mode: Optional[str] = None,
) -> WandbTracker:
    config = {
        "pipeline": "exp_july",
        "video_ids": list(video_ids) if video_ids is not None else None,
        "video_count": video_count,
        "rounds": rounds,
        "max_step": max_step,
        "step8_pattern_llm_model": os.environ.get(
            "CAUVID_STEP8_PATTERN_LLM_MODEL",
            os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        ),
        "step8c_llm_timeout_seconds": _env_float(
            "CAUVID_STEP8C_LLM_TIMEOUT_SECONDS", 120.0
        ),
        "step8c_llm_max_attempts": _env_int(
            "CAUVID_STEP8C_LLM_MAX_ATTEMPTS", 3, minimum=1
        ),
        "step8c_llm_retry_backoff_seconds": _env_float(
            "CAUVID_STEP8C_LLM_RETRY_BACKOFF_SECONDS", 2.0
        ),
        "step8c_review_interval_tracks": _env_int(
            "CAUVID_STEP8C_REVIEW_INTERVAL_TRACKS", 500, minimum=1
        ),
        "step8_threshold_min_conflicts": _env_int(
            "CAUVID_STEP8_THRESHOLD_MIN_CONFLICTS", 10, minimum=1
        ),
        "step8_threshold_target_quantile": _env_float(
            "CAUVID_STEP8_THRESHOLD_TARGET_QUANTILE", 0.9
        ),
        "step8_threshold_max_relative_change": _env_float(
            "CAUVID_STEP8_THRESHOLD_MAX_RELATIVE_CHANGE", 0.1
        ),
        "step8_threshold_max_unprotected_flip_rate": _env_float(
            "CAUVID_STEP8_THRESHOLD_MAX_UNPROTECTED_FLIP_RATE", 0.02
        ),
        "step8_threshold_validation_fraction": _env_float(
            "CAUVID_STEP8_THRESHOLD_VALIDATION_FRACTION", 0.2
        ),
    }
    return WandbTracker(
        config=config,
        enabled=enabled,
        project=project,
        run_name=run_name,
        mode=mode,
    )
