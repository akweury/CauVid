import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exp_july.perception import step1_init
from src.exp_july.perception import step2_detection
from src.exp_july.perception import step3_tracking
from src.exp_july.perception import step6_positions_3d
from src.exp_july.perception import step7_ego_motion
from src.exp_july.perception import step7b_tracklet_repair
from src.exp_july.perception import step8_relative_object_motion
from src.exp_july.perception import step8a_symbol_grounded_refinement
from src.exp_july.perception import step8a_visual_symbol_grounded
from src.exp_july.perception import step8_visual_relative_motion
from src.exp_july.perception import step8b_causal_filter_out
from src.exp_july.perception import step8c_prior_guided_ego_motion_refinement
from src.exp_july.perception import step8d_adaptive_protected_object_motion_repair
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


def main(video_ids=None, video_count=None, rounds=3, max_step=18):
    # Step 1: initialize dataset scope and selected videos.
    env = step1_init(video_ids=video_ids, video_count=video_count)
    if max_step <= 1:
        return env
    # Step 2: prepare detection outputs.
    detection_state = step2_detection(env, env["detection_args"])
    if max_step <= 2:
        return detection_state
    # Step 3: build object tracks from detections.
    tracking_state = step3_tracking(detection_state)
    if max_step <= 3:
        return tracking_state
    # Step 4-5: removed; downstream uses OD detections and tracks only.
    if max_step <= 5:
        return tracking_state
    # Step 6: prepare 3D positions or geometry.
    position_state = step6_positions_3d(tracking_state)
    if max_step <= 6:
        return position_state
    # Step 7: estimate ego motion signals.
    ego_state = step7_ego_motion(position_state)
    if max_step <= 7:
        return ego_state
    # Step 7B: repair broken tracklets after ego estimation.
    repaired_state = step7b_tracklet_repair(position_state, ego_state)
    # Step 8: compute relative object motion.
    relative_motion_state = step8_relative_object_motion(position_state, repaired_state)
    # Step 8A: generate grounded semantic protection rules and apply the protection prior.
    relative_motion_state = step8a_symbol_grounded_refinement(relative_motion_state)
    relative_motion_state = step8a_visual_symbol_grounded(relative_motion_state)
    # Step 8B: validate trajectory motion facts and decide symbolic-layer eligibility.
    relative_motion_state = step8b_causal_filter_out(ego_state, relative_motion_state)
    # Step 8C: select reliable static/low-dynamic references for future ego refinement.
    relative_motion_state = step8c_prior_guided_ego_motion_refinement(ego_state, relative_motion_state)
    # Step 8D: diagnostically repair protected tracks rejected by 8B.
    relative_motion_state = step8d_adaptive_protected_object_motion_repair(relative_motion_state)
    # Step 8 visual: render per-track relative motion videos with causal filter decisions.
    relative_motion_state = step8_visual_relative_motion(relative_motion_state)
    if max_step <= 8:
        return relative_motion_state
    # Step 9: segment videos into temporal chunks.
    segment_state = step9_temporal_segmentation(ego_state, relative_motion_state)
    if max_step <= 9:
        return segment_state

    # Step 10: summarize object motion per segment.
    segment_motion_state = step10_segment_object_motion(segment_state)
    if max_step <= 10:
        return segment_motion_state
    # Step 11: select important objects for reasoning.
    important_object_state = step11_important_objects(segment_motion_state)
    if max_step <= 11:
        return important_object_state
    # Step 12: convert scene summaries into logic atoms.
    atom_state = step12_logic_atoms(important_object_state)
    if max_step <= 12:
        return atom_state
    # Step 13: define target heads for rule learning.
    target_head_state = step13_target_heads(atom_state)
    if max_step <= 13:
        return target_head_state
    # Step 14: build temporal rule-learning examples.
    example_state = step14_temporal_rule_examples(atom_state, target_head_state)
    if max_step <= 14:
        return example_state
    # Step 15: mine candidate rules from examples.
    candidate_rule_state = step15_candidate_rules(example_state)
    if max_step <= 15:
        return candidate_rule_state
    # Step 16: merge and extend the rule pool.
    rule_pool_state = step16_merge_and_extend_rules(candidate_rule_state)
    if max_step <= 16:
        return rule_pool_state
    # Step 17: select the initial final rule set.
    selection_state = step17_final_rule_selection(rule_pool_state)
    if max_step <= 17:
        return selection_state
    # Step 18: run causal refinement with iterative rounds.
    refined_state = step18_causal_refinement(selection_state, rounds=rounds)
    return refined_state


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the exp_july pipeline locally")
    parser.add_argument("--video-ids", nargs="*", default=None, help="Specific video IDs to process")
    parser.add_argument("--video-count", type=int, default=None, help="Limit the run to this many videos")
    parser.add_argument("--rounds", type=int, default=3, help="Number of causal refinement rounds")
    parser.add_argument("--max-step", type=int, default=18, help="Highest pipeline step to execute")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = main(
        video_ids=args.video_ids,
        video_count=args.video_count,
        rounds=args.rounds,
        max_step=args.max_step,
    )
    print("done")
    print(f"videos={len(result['videos'])}")
    if "rounds" in result:
        print(f"rounds={len(result['rounds'])}")
