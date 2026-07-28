import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exp_july.perception import step1_init
from src.exp_july.perception import step2_detection
from src.exp_july.perception import step3_tracking
from src.exp_july.perception import step6_positions_3d
from src.exp_july.perception import step7_train_eval_split
from src.exp_july.perception import step7a_axis_threshold_segmentation
from src.exp_july.perception import step8_trajectory_repair
from src.exp_july.perception import step8a_relative_object_motion
from src.exp_july.perception import step8b_signal_evidence
from src.exp_july.perception import step8c_trajectory_clustering
from src.exp_july.perception import step8d_closed_loop_trajectory_repair
from src.exp_july.perception import step8e_repaired_trajectory_validation
from src.exp_july.perception import step8f_trajectory_statistics
from src.exp_july.perception import step8g_repaired_track_materialization
from src.exp_july.perception import step8h_trajectory_repair_visualization
from src.exp_july.perception import step8i_trajectory_audit_dashboard
from src.exp_july.perception import step8j_trajectory_provenance_audit
from src.exp_july.perception import step8k_trajectory_handoff
from src.exp_july.perception import step9_temporal_segmentation
from src.exp_july.perception import step10_segment_object_motion


def step11_important_objects(segment_motion_state):
    return {"videos": segment_motion_state["videos"], "important_objects": []}


def step12_logic_atoms(important_object_state):
    return {"videos": important_object_state["videos"], "logic_atoms": []}


def step13_target_heads(atom_state):
    return {"videos": atom_state["videos"], "target_heads": []}


def step14_temporal_rule_examples(atom_state, target_head_state):
    return {"videos": atom_state["videos"], "temporal_rule_examples": []}


def step15_candidate_rules(example_state):
    return {"videos": example_state["videos"], "candidate_rules": []}


def step16_merge_and_extend_rules(candidate_rule_state):
    return {
        "videos": candidate_rule_state["videos"],
        "merged_rules": [],
        "extended_rules": [],
        "ranked_rules": [],
    }


def step17_final_rule_selection(rule_pool_state):
    return {
        "videos": rule_pool_state["videos"],
        "ranked_rules": rule_pool_state["ranked_rules"],
        "final_rules": [],
        "top_k": 0,
    }


def step18_causal_refinement(selection_state, rounds=3):
    active_rules = selection_state["final_rules"]
    ranked_rules = selection_state["ranked_rules"]
    history = []
    for round_idx in range(rounds):
        step18_eval = {"round": round_idx + 1, "active_rules": active_rules}
        step18m_masking = {"round": round_idx + 1, "causal_effects": []}
        step18n_reselection = {
            "round": round_idx + 1,
            "removed_rules": [],
            "added_rules": [],
            "active_rules": active_rules,
            "ranked_rules": ranked_rules,
        }
        step18o_refined_eval = {"round": round_idx + 1, "refined_rules": step18n_reselection["active_rules"]}
        active_rules = step18n_reselection["active_rules"]
        history.append(
            {
                "step18": step18_eval,
                "step18m": step18m_masking,
                "step18n": step18n_reselection,
                "step18o": step18o_refined_eval,
            }
        )
    return {"videos": selection_state["videos"], "refined_final_rules": active_rules, "rounds": history}


def _sum_fields(rows, *field_names):
    total = 0
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for field_name in field_names:
            try:
                total += int(row.get(field_name, 0) or 0)
            except (TypeError, ValueError):
                continue
    return total


def _step_data_error(step_name, state):
    """Return a clear error when a completed stage has no processable data."""
    if not isinstance(state, dict):
        return f"returned {type(state).__name__}; expected a state dictionary"

    videos = state.get("videos", [])
    if not videos:
        detail = ""
        if step_name == "01_init":
            dataset_root = state.get("dataset_root")
            detail = (
                f" under dataset_root={dataset_root}; expected either "
                f"{dataset_root}/frames/<video_id>/frame_*.jpg or "
                f"{dataset_root}/videos/<video_id>.(mov|mp4|avi|mkv); "
                "check CAUVID_DRIVING_MINI_HOST and requested video IDs"
            )
        return f"produced no videos{detail}"

    if step_name == "02_detection":
        detections = state.get("detections", [])
        if not detections:
            return "produced no per-video detection results"
        if _sum_fields(
            detections, "num_detections", "num_candidate_detections"
        ) <= 0:
            return "produced zero object detections"

    if step_name == "03_tracking":
        tracks = state.get("tracks", [])
        if not tracks:
            return "produced no per-video tracking results"
        if _sum_fields(tracks, "num_tracks", "num_candidate_tracks") <= 0:
            return "produced zero object tracks"

    if step_name == "06_positions_3d":
        positions = state.get("positions_3d", [])
        if not positions:
            return "produced no per-video 3D-position results"
        if _sum_fields(
            positions,
            "num_objects_with_3d",
            "num_candidate_objects_with_3d",
        ) <= 0:
            return "produced zero objects with 3D positions"

    if step_name == "07_ego_motion":
        ego_motion = state.get("ego_motion", [])
        if not ego_motion:
            return "produced no per-video ego-motion results"
        if _sum_fields(ego_motion, "num_frames_with_ego_motion") <= 0:
            return "produced zero frames with ego-motion estimates"

    if step_name == "07a_ego_symbol_prior":
        manifest = state.get("ego_symbol_prior_manifest", {})
        if int(manifest.get("num_videos", 0) or 0) <= 0:
            return "produced no ego-symbol-prior videos"
        if int(manifest.get("num_frames", 0) or 0) <= 0:
            return "produced zero ego-symbol-prior frames"

    if step_name == "07a_axis_threshold_segmentation":
        manifest = state.get("ego_axis_threshold_segmentation_manifest", {})
        if int(manifest.get("num_videos", 0) or 0) <= 0:
            return "produced no axis-threshold-segmentation videos"
        if int(manifest.get("num_frames", 0) or 0) <= 0:
            return "produced zero axis-threshold-segmentation frames"

    if step_name == "08_trajectory_repair":
        repaired = state.get("positions_3d", state.get("tracklet_repair", []))
        if not repaired:
            return "produced no repaired per-video position results"
        if _sum_fields(
            repaired,
            "num_objects_with_3d",
            "num_candidate_objects_with_3d",
            "num_objects_total",
        ) <= 0:
            return "produced zero repaired object-position observations"

    if step_name == "08a_relative_motion":
        relative_motion = state.get("relative_object_motion", [])
        if not relative_motion:
            return "contains no per-video relative-motion results"
        if _sum_fields(
            relative_motion,
            "num_objects_with_rel_motion",
            "num_objects_total",
        ) <= 0:
            return "contains zero relative-motion object tracks"

    if step_name == "08b_uncertain_signal_evidence":
        manifest = state.get("uncertain_signal_evidence_manifest", {})
        if int(manifest.get("num_videos", 0) or 0) <= 0:
            return "produced no signal-evidence videos"
        if int(manifest.get("num_tracks", 0) or 0) <= 0:
            return "produced zero signal-evidence tracks"
        if int(manifest.get("num_observations", 0) or 0) <= 0:
            return "produced zero signal-evidence observations"

    if step_name == "08c_trajectory_clustering":
        manifest = state.get("trajectory_clustering_manifest", {})
        if int(manifest.get("num_videos", 0) or 0) <= 0:
            return "clustered no trajectory videos"
        if int(manifest.get("num_tracks", 0) or 0) <= 0:
            return "clustered zero trajectory tracks"

    if step_name == "08d_closed_loop_trajectory_repair":
        manifest = state.get("trajectory_pattern_manifest", {})
        if int(manifest.get("num_videos", 0) or 0) <= 0:
            return "repaired no trajectory videos"
        if int(manifest.get("num_tracks", 0) or 0) <= 0:
            return "processed zero tracks for closed-loop repair"

    if step_name == "08e_repaired_trajectory_validation":
        manifest = state.get("step8e_validation_manifest", {})
        if int(manifest.get("num_tracks", 0) or 0) <= 0:
            return "published zero repaired-trajectory validation records"

    required_collections = {
        "09_temporal_segmentation": (
            "temporal_segments",
            "produced no temporal segments",
        ),
        "10_segment_motion": (
            "segment_object_motion",
            "produced no segment-level object motion",
        ),
        "11_important_objects": (
            "important_objects",
            "selected no important objects",
        ),
        "12_logic_atoms": ("logic_atoms", "produced no logic atoms"),
        "13_target_heads": ("target_heads", "produced no target heads"),
        "14_rule_examples": (
            "temporal_rule_examples",
            "produced no temporal rule examples",
        ),
        "15_candidate_rules": (
            "candidate_rules",
            "produced no candidate rules",
        ),
        "16_rule_pool": ("ranked_rules", "produced an empty rule pool"),
        "17_rule_selection": ("final_rules", "selected no final rules"),
        "18_causal_refinement": (
            "rounds",
            "produced no causal-refinement rounds",
        ),
    }
    required = required_collections.get(step_name)
    if required is not None:
        field_name, message = required
        if not state.get(field_name):
            return message

    return None


def _require_step_data(step_name, state):
    error = _step_data_error(step_name, state)
    if error is None:
        return state
    message = f"[pipeline][error] {step_name}: {error}; stopping pipeline"
    print(message, file=sys.stderr, flush=True)
    raise RuntimeError(message)


def _tracked_step(tracker, step_name, operation):
    started = time.perf_counter()
    try:
        state = operation()
        _require_step_data(step_name, state)
    except BaseException as exc:
        tracker.log_failure(
            step_name,
            exc,
            duration_seconds=time.perf_counter() - started,
        )
        raise
    return tracker.log_state(
        step_name,
        state,
        duration_seconds=time.perf_counter() - started,
    )


def _run_pipeline(
    video_ids,
    video_count,
    rounds,
    max_step,
    tracker,
    step7_profile="eval-fast",
    step7_threshold_search_rounds=3,
    step7e_expensive_candidate_limit=8,
):
    # Step 1: initialize dataset scope and selected videos.
    env = _tracked_step(
        tracker,
        "01_init",
        lambda: step1_init(video_ids=video_ids, video_count=video_count),
    )
    if max_step <= 1:
        return env
    # Step 2: prepare detection outputs.
    detection_state = _tracked_step(
        tracker,
        "02_detection",
        lambda: step2_detection(env, env["detection_args"]),
    )
    if max_step <= 2:
        return detection_state
    # Step 3: build object tracks from detections.
    tracking_state = _tracked_step(
        tracker, "03_tracking", lambda: step3_tracking(detection_state)
    )
    if max_step <= 3:
        return tracking_state
    # Step 4-5: removed; downstream uses OD detections and tracks only.
    if max_step <= 5:
        return tracking_state
    # Step 6: prepare 3D positions or geometry.
    position_state = _tracked_step(
        tracker, "06_positions_3d", lambda: step6_positions_3d(tracking_state)
    )
    if max_step <= 6:
        return position_state
    # Step 7: intentionally empty. The former 7/7A-7F ego-motion pipeline is
    # archived but not executed. Preserve the Step 6 payload for Step 8 and
    # expose explicit empty fields so downstream consumers never reuse stale
    # ego-motion or ego-symbol results.
    ego_final_state = {
        **position_state,
        "step7_status": "empty",
        "step7_substeps": [],
        "ego_motion": [],
        "ego_symbol_prior": [],
        "final_ego_symbols": [],
    }
    if max_step <= 7:
        return ego_final_state
    # Split videos before Step 7A. Density is fitted on train videos and
    # evaluated only on held-out evaluation videos.
    step7_split_state = _tracked_step(
        tracker,
        "07_train_eval_split",
        lambda: step7_train_eval_split(position_state),
    )
    # Step 7A: the only active Step 7 analysis substep.
    ego_final_state = _tracked_step(
        tracker,
        "07a_axis_threshold_segmentation",
        lambda: step7a_axis_threshold_segmentation(step7_split_state),
    )
    # Step 8: repair trajectories first; split events receive new track IDs.
    repaired_state = _tracked_step(
        tracker,
        "08_trajectory_repair",
        lambda: step8_trajectory_repair(position_state, ego_final_state),
    )
    # Step 8A: compute relative motion from the repaired, canonical track IDs.
    relative_motion_state = _tracked_step(
        tracker,
        "08a_relative_motion",
        lambda: step8a_relative_object_motion(position_state, repaired_state),
    )
    # Legacy threshold-epoch activation is archived and intentionally disabled.
    # Step 8B: abstract uncertain position/vx/vz signals without classifying motion.
    relative_motion_state = _tracked_step(
        tracker,
        "08b_uncertain_signal_evidence",
        lambda: step8b_signal_evidence(relative_motion_state),
    )
    # Step 8C: symbolic trajectory abstraction and cohort assignment only.
    relative_motion_state = _tracked_step(
        tracker,
        "08c_trajectory_clustering",
        lambda: step8c_trajectory_clustering(relative_motion_state),
    )
    # Step 8D: deterministic closed-loop repair using frozen Step 8C cohorts.
    relative_motion_state = _tracked_step(
        tracker,
        "08d_closed_loop_trajectory_repair",
        lambda: step8d_closed_loop_trajectory_repair(relative_motion_state),
    )
    # Step 8E: publish repaired-trajectory validation outcomes.
    relative_motion_state = _tracked_step(
        tracker,
        "08e_repaired_trajectory_validation",
        lambda: step8e_repaired_trajectory_validation(relative_motion_state),
    )
    # Step 8F: publish versioned statistical aggregation and promotion results.
    relative_motion_state = _tracked_step(
        tracker,
        "08f_trajectory_statistics",
        lambda: step8f_trajectory_statistics(relative_motion_state),
    )
    # Step 8G: checkpoint repaired tracks for downstream use.
    relative_motion_state = _tracked_step(
        tracker,
        "08g_repaired_track_materialization",
        lambda: step8g_repaired_track_materialization(relative_motion_state),
    )
    # Step 8H: render comparison videos, HTML reports, and statistical PDFs.
    relative_motion_state = _tracked_step(
        tracker,
        "08h_trajectory_repair_visualization",
        lambda: step8h_trajectory_repair_visualization(relative_motion_state),
    )
    # Step 8I: build the offline read-only audit dashboard.
    relative_motion_state = _tracked_step(
        tracker,
        "08i_trajectory_audit_dashboard",
        lambda: step8i_trajectory_audit_dashboard(relative_motion_state),
    )
    # Step 8J: persist cross-stage provenance.
    relative_motion_state = _tracked_step(
        tracker,
        "08j_trajectory_provenance_audit",
        lambda: step8j_trajectory_provenance_audit(relative_motion_state),
    )
    # Step 8K: finalize the new Step 8 branch for downstream stages.
    relative_motion_state = _tracked_step(
        tracker,
        "08k_trajectory_handoff",
        lambda: step8k_trajectory_handoff(relative_motion_state),
    )
    if max_step <= 8:
        return relative_motion_state
    # Step 9: segment videos into temporal chunks.
    segment_state = _tracked_step(
        tracker,
        "09_temporal_segmentation",
        lambda: step9_temporal_segmentation(ego_state, relative_motion_state),
    )
    if max_step <= 9:
        return segment_state

    # Step 10: summarize object motion per segment.
    segment_motion_state = _tracked_step(
        tracker,
        "10_segment_motion",
        lambda: step10_segment_object_motion(segment_state),
    )
    if max_step <= 10:
        return segment_motion_state
    # Step 11: select important objects for reasoning.
    important_object_state = _tracked_step(
        tracker,
        "11_important_objects",
        lambda: step11_important_objects(segment_motion_state),
    )
    if max_step <= 11:
        return important_object_state
    # Step 12: convert scene summaries into logic atoms.
    atom_state = _tracked_step(
        tracker, "12_logic_atoms", lambda: step12_logic_atoms(important_object_state)
    )
    if max_step <= 12:
        return atom_state
    # Step 13: define target heads for rule learning.
    target_head_state = _tracked_step(
        tracker,
        "13_target_heads",
        lambda: step13_target_heads(atom_state),
    )
    if max_step <= 13:
        return target_head_state
    # Step 14: build temporal rule-learning examples.
    example_state = _tracked_step(
        tracker,
        "14_rule_examples",
        lambda: step14_temporal_rule_examples(atom_state, target_head_state),
    )
    if max_step <= 14:
        return example_state
    # Step 15: mine candidate rules from examples.
    candidate_rule_state = _tracked_step(
        tracker,
        "15_candidate_rules",
        lambda: step15_candidate_rules(example_state),
    )
    if max_step <= 15:
        return candidate_rule_state
    # Step 16: merge and extend the rule pool.
    rule_pool_state = _tracked_step(
        tracker,
        "16_rule_pool",
        lambda: step16_merge_and_extend_rules(candidate_rule_state),
    )
    if max_step <= 16:
        return rule_pool_state
    # Step 17: select the initial final rule set.
    selection_state = _tracked_step(
        tracker,
        "17_rule_selection",
        lambda: step17_final_rule_selection(rule_pool_state),
    )
    if max_step <= 17:
        return selection_state
    # Step 18: run causal refinement with iterative rounds.
    refined_state = _tracked_step(
        tracker,
        "18_causal_refinement",
        lambda: step18_causal_refinement(selection_state, rounds=rounds),
    )
    return refined_state


def main(
    video_ids=None,
    video_count=None,
    rounds=3,
    max_step=18,
    step7_profile="eval-fast",
    step7_threshold_search_rounds=3,
    step7e_expensive_candidate_limit=8,
    *,
    wandb_enabled=None,
    wandb_project=None,
    wandb_run_name=None,
    wandb_mode=None,
):
    from src.exp_july.wandb_tracking import create_tracker

    if isinstance(video_ids, str):
        video_ids = [video_ids]
    elif video_ids is not None and not isinstance(video_ids, list):
        video_ids = list(video_ids)
    tracker = create_tracker(
        video_ids=video_ids,
        video_count=video_count,
        rounds=rounds,
        max_step=max_step,
        enabled=wandb_enabled,
        project=wandb_project,
        run_name=wandb_run_name,
        mode=wandb_mode,
    )
    try:
        result = _run_pipeline(
            video_ids, video_count, rounds, max_step, tracker,
            step7_profile=step7_profile,
            step7_threshold_search_rounds=step7_threshold_search_rounds,
            step7e_expensive_candidate_limit=step7e_expensive_candidate_limit,
        )
    except BaseException as exc:
        tracker.finish(status="failed", error=exc)
        raise
    tracker.finish(status="completed")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the exp_july pipeline locally")
    parser.add_argument("--video-ids", nargs="*", default=None, help="Specific video IDs to process")
    parser.add_argument("--video-count", type=int, default=None, help="Limit the run to this many videos")
    parser.add_argument("--rounds", type=int, default=3, help="Number of causal refinement rounds")
    parser.add_argument("--max-step", type=int, default=18, help="Highest pipeline step to execute")
    parser.add_argument(
        "--step7-profile",
        choices=("eval-fast", "train"),
        default="eval-fast",
        help="Deprecated compatibility option; Step 7 is currently empty",
    )
    parser.add_argument(
        "--step7-threshold-search-rounds",
        type=int,
        default=3,
        help="Deprecated compatibility option; Step 7 is currently empty",
    )
    parser.add_argument(
        "--step7e-expensive-candidate-limit",
        type=int,
        default=8,
        help="Deprecated compatibility option; Step 7 is currently empty",
    )
    wandb_group = parser.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", dest="wandb_enabled", action="store_true", help="Enable W&B tracking")
    wandb_group.add_argument("--no-wandb", dest="wandb_enabled", action="store_false", help="Disable W&B tracking")
    parser.set_defaults(wandb_enabled=None)
    parser.add_argument("--wandb-project", default=None, help="W&B project override")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = main(
        video_ids=args.video_ids,
        video_count=args.video_count,
        rounds=args.rounds,
        max_step=args.max_step,
        step7_profile=args.step7_profile,
        step7_threshold_search_rounds=max(1, args.step7_threshold_search_rounds),
        step7e_expensive_candidate_limit=max(1, args.step7e_expensive_candidate_limit),
        wandb_enabled=args.wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
    )
    print("done")
    print(f"videos={len(result['videos'])}")
    if "rounds" in result:
        print(f"rounds={len(result['rounds'])}")
