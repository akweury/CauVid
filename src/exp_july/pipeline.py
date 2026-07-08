import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.exp_driving_videos import pipeline_config as driving_pipeline_config
from src.exp_driving_videos.modules import detect_driving_mini


def get_pipeline_output_root():
    return Path(os.environ.get("CAUVID_PIPELINE_OUTPUT_PATH", ROOT / "output_july"))


def step1_init(video_ids=None, video_count=None):
    dataset_root = config.get_dataset_path("driving_mini")
    video_dir = dataset_root / "videos"
    all_videos = sorted(config.get_mini_video_ids()) if video_dir.exists() else []
    if video_ids:
        videos = []
        for video_id in video_ids:
            if video_id and video_id not in videos:
                videos.append(video_id)
    else:
        videos = list(all_videos)
    if video_count is not None:
        videos = videos[:video_count]
    print(
        f"[step 1] loaded {len(videos)} videos for this run \n"
        f"[step 1] from {video_dir} \n"
        f"[step 1] (dataset=driving_mini, total_in_dataset={len(all_videos)})"
    )
    detection_args = {
        "video_ids": videos,
        "model_name": driving_pipeline_config.DRIVING_MINI_OD_MODEL,
        "classes": driving_pipeline_config.DRIVING_MINI_OD_CLASSES,
        "output_root": get_pipeline_output_root() / "01_driving_mini_detection",
        "od_calibration_policy": {},
        "force_recompute": False,
        "render_video": driving_pipeline_config.get_detection_render_video_enabled(default=True),
        "check_cache": driving_pipeline_config.get_detection_check_cache_enabled(default=False),
        "enable_candidate_branch": driving_pipeline_config.get_detection_candidate_branch_enabled(default=False),
        "skip_step": driving_pipeline_config.get_detection_skip_step_enabled(default=False),
    }
    return {"videos": videos, "dataset_root": dataset_root, "detection_args": detection_args}


def step2_detection(env, args):
    videos = env["videos"]
    if not videos:
        print("[step 2] no videos selected, skip detection")
        return {"videos": [], "detections": [], "detection_output_root": None}

    run_args = dict(args)
    model_name = run_args["model_name"]
    classes = run_args["classes"]
    skip_step = bool(run_args.pop("skip_step", False))
    render_video = bool(run_args.get("render_video", True))
    check_cache = bool(run_args.get("check_cache", False))
    candidate_branch_enabled = bool(run_args.get("enable_candidate_branch", False))
    output_root = Path(run_args["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[step 2] model={model_name}")
    print(f"[step 2] classes={len(classes)} render_video={render_video} check_cache={check_cache}")
    print(f"[step 2] output_root={output_root}")

    if skip_step:
        manifest_path = output_root / "detection_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        video_entries = {
            str(entry.get("video_id", "")).strip(): dict(entry)
            for entry in manifest.get("videos", [])
            if str(entry.get("video_id", "")).strip()
        }
        detections = []
        for video_id in videos:
            entry = video_entries[video_id]
            detections_path = Path(str(entry.get("detections_json", "")).strip() or output_root / video_id / "detections.json")
            with detections_path.open("r", encoding="utf-8") as f:
                video_result = json.load(f)
            if hasattr(detect_driving_mini, "_apply_candidate_branch_mode"):
                video_result = detect_driving_mini._apply_candidate_branch_mode(video_result, candidate_branch_enabled)
            detections.append(video_result)
        print(f"[step 2] loaded cached detection results for {len(detections)} videos")
    else:
        detections = detect_driving_mini.run(**run_args)
        print(f"[step 2] completed detection for {len(detections)} videos")

    accepted_count = sum(int(video_result.get("num_detections", 0)) for video_result in detections)
    candidate_count = sum(int(video_result.get("num_candidate_detections", 0)) for video_result in detections)
    print(f"[step 2] accepted_detections={accepted_count}, candidate_detections={candidate_count}")
    return {
        "videos": videos,
        "detections": detections,
        "detection_output_root": output_root,
        "model_name": model_name,
        "classes": classes,
    }


def step3_tracking(detection_state):
    return {"videos": detection_state["videos"], "tracks": []}


def step4_dataset_annotations(env):
    return {"videos": env["videos"], "dataset_annotations": []}


def step5_merge_annotations(tracking_state, annotation_state):
    return {"videos": tracking_state["videos"], "merged_annotations": []}


def step6_positions_3d(merged_state):
    return {"videos": merged_state["videos"], "positions_3d": []}


def step7_ego_motion(position_state):
    return {"videos": position_state["videos"], "ego_motion": []}


def step8_relative_object_motion(position_state, ego_state):
    return {"videos": position_state["videos"], "relative_object_motion": []}


def step9_temporal_segmentation(ego_state, relative_motion_state):
    return {"videos": ego_state["videos"], "temporal_segments": []}


def step10_segment_object_motion(segment_state):
    return {"videos": segment_state["videos"], "segment_object_motion": []}


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


def main(video_ids=None, video_count=None, rounds=3):
    # Step 1: initialize dataset scope and selected videos.
    env = step1_init(video_ids=video_ids, video_count=video_count)
    # Step 2: prepare detection outputs.
    detection_state = step2_detection(env, env["detection_args"])
    # Step 3: build object tracks from detections.
    tracking_state = step3_tracking(detection_state)
    # Step 4: load dataset annotations.
    annotation_state = step4_dataset_annotations(env)
    # Step 5: merge tracking results with annotations.
    merged_state = step5_merge_annotations(tracking_state, annotation_state)
    # Step 6: prepare 3D positions or geometry.
    position_state = step6_positions_3d(merged_state)
    # Step 7: estimate ego motion signals.
    ego_state = step7_ego_motion(position_state)
    # Step 8: compute relative object motion.
    relative_motion_state = step8_relative_object_motion(position_state, ego_state)
    # Step 9: segment videos into temporal chunks.
    segment_state = step9_temporal_segmentation(ego_state, relative_motion_state)
    # Step 10: summarize object motion per segment.
    segment_motion_state = step10_segment_object_motion(segment_state)
    # Step 11: select important objects for reasoning.
    important_object_state = step11_important_objects(segment_motion_state)
    # Step 12: convert scene summaries into logic atoms.
    atom_state = step12_logic_atoms(important_object_state)
    # Step 13: define target heads for rule learning.
    target_head_state = step13_target_heads(atom_state)
    # Step 14: build temporal rule-learning examples.
    example_state = step14_temporal_rule_examples(atom_state, target_head_state)
    # Step 15: mine candidate rules from examples.
    candidate_rule_state = step15_candidate_rules(example_state)
    # Step 16: merge and extend the rule pool.
    rule_pool_state = step16_merge_and_extend_rules(candidate_rule_state)
    # Step 17: select the initial final rule set.
    selection_state = step17_final_rule_selection(rule_pool_state)
    # Step 18: run causal refinement with iterative rounds.
    refined_state = step18_causal_refinement(selection_state, rounds=rounds)
    return refined_state


if __name__ == "__main__":
    result = main()
    print("done")
    print(f"videos={len(result['videos'])}")
    print(f"rounds={len(result['rounds'])}")
