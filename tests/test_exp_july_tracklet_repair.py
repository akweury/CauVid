import importlib.util
import sys
import types
from pathlib import Path


def _stub_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


_stub_module(
    "src.exp_driving_videos.pipeline_config",
    DRIVING_MINI_OD_MODEL="stub",
    DRIVING_MINI_OD_CLASSES=[],
    get_detection_render_video_enabled=lambda default=True: default,
    get_detection_check_cache_enabled=lambda default=False: default,
    get_detection_candidate_branch_enabled=lambda default=False: default,
    get_detection_skip_step_enabled=lambda default=False: default,
    get_tracking_render_video_enabled=lambda default=True: default,
    get_ego_motion_smoothing_window=lambda default=5: default,
    get_ego_static_adjustment_cfg=lambda: {},
    get_ego_motion_render_video_enabled=lambda default=True: default,
)
_stub_module("src.exp_driving_videos.modules.detect_driving_mini")
_stub_module(
    "src.exp_driving_videos.modules.ego_motion_driving_mini",
    _EGO_MOTION_VERSION=2,
    _EGO_MOTION_METHOD="stub",
)
_stub_module("src.exp_driving_videos.modules.merge_gt_and_detected_driving_mini")
_stub_module("src.exp_driving_videos.modules.prepare_3d_positions_driving_mini", _POSITIONS_3D_VERSION=4)
_stub_module(
    "src.exp_driving_videos.modules.tracking_driving_mini",
    cv2=None,
    _TRACKING_SCHEMA_VERSION=7,
    ensure_tracking_runtime_available=lambda: None,
)

_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "src" / "exp_july" / "perception" / "pipeline.py"
_SPEC = importlib.util.spec_from_file_location("exp_july_perception_pipeline_under_test", _PIPELINE_PATH)
_PIPELINE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PIPELINE)
_repair_video_tracklets = _PIPELINE._repair_video_tracklets
_relative_motion_video = _PIPELINE._relative_motion_video
_trajectory_motion_evidence_video = _PIPELINE._trajectory_motion_evidence_video


def _frame(frame_index, boxes=None, scores=None, labels=None, track_ids=None, positions_3d=None):
    boxes = boxes or []
    scores = scores or []
    labels = labels or []
    track_ids = track_ids or []
    positions_3d = positions_3d or []
    return {
        "frame": f"frame_{frame_index:05d}.jpg",
        "frame_index": frame_index,
        "image_path": f"/tmp/frame_{frame_index:05d}.jpg",
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
        "track_ids": track_ids,
        "sources": ["det" for _ in boxes],
        "objects": [
            {
                "bbox": box,
                "score": score,
                "label": label,
                "track_id": track_id,
                "accepted": True,
                "source": "det",
                "source_type": "accepted_track",
                "is_ground_truth": False,
            }
            for box, score, label, track_id in zip(boxes, scores, labels, track_ids)
        ],
        "positions_3d": positions_3d,
        "candidate_positions_3d": [],
        "has_3d_positions": bool(positions_3d),
    }


def test_tracklet_repair_interpolates_safe_one_frame_gap():
    video = {
        "video_id": "vid",
        "num_frames": 3,
        "num_objects": 2,
        "num_objects_with_3d": 2,
        "frames": [
            _frame(0, [[0, 0, 10, 10]], [0.8], ["car"], [7], [[0, 0, 10]]),
            _frame(1),
            _frame(2, [[10, 0, 20, 10]], [0.7], ["car"], [7], [[2, 0, 10]]),
        ],
    }

    repaired = _repair_video_tracklets(video, {"video_id": "vid"})

    assert repaired["tracklet_repair"]["num_repaired_gaps"] == 1
    assert repaired["tracklet_repair"]["num_interpolated_objects"] == 1
    frame = repaired["frames"][1]
    assert frame["boxes"] == [[5.0, 0.0, 15.0, 10.0]]
    assert frame["positions_3d"] == [[1.0, 0.0, 10.0]]
    assert frame["labels"] == ["car"]
    assert frame["track_ids"] == [7]
    assert frame["objects"][0]["source_type"] == "interpolated_tracklet"


def test_tracklet_repair_rejects_overlap_conflict():
    video = {
        "video_id": "vid",
        "num_frames": 3,
        "num_objects": 3,
        "num_objects_with_3d": 3,
        "frames": [
            _frame(0, [[0, 0, 10, 10]], [0.8], ["car"], [7], [[0, 0, 10]]),
            _frame(1, [[5, 0, 15, 10]], [0.9], ["car"], [8], [[1, 0, 10]]),
            _frame(2, [[10, 0, 20, 10]], [0.7], ["car"], [7], [[2, 0, 10]]),
        ],
    }

    repaired = _repair_video_tracklets(video, {"video_id": "vid"})

    assert repaired["tracklet_repair"]["num_repaired_gaps"] == 0
    assert repaired["tracklet_repair"]["skipped_gaps"][0]["reason"] == "overlap_conflict"
    assert repaired["frames"][1]["track_ids"] == [8]


def test_relative_motion_marks_observed_and_repaired_sources_with_frame_labels():
    video = {
        "video_id": "vid",
        "frames": [
            _frame(0, [[0, 0, 10, 10]], [0.8], ["car"], [7], [[0, 0, 10]]),
            _frame(1, [[5, 0, 15, 10]], [0.7], ["car"], [7], [[1, 0, 10]]),
        ],
    }
    video["frames"][1]["sources"] = ["tracklet_repair"]
    video["frames"][1]["objects"][0]["source"] = "tracklet_repair"
    video["frames"][1]["objects"][0]["source_type"] = "interpolated_tracklet"
    video["frames"][1]["objects"][0]["position_3d"] = [1, 0, 10]
    video["frames"][1]["objects"][0]["repair_provenance"] = {"method": "bounded_linear_interpolation"}
    ego = {
        "video_id": "vid",
        "frames": [
            {"frame_index": 0, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
            {"frame_index": 1, "ego_vx_smoothed": 0.1, "ego_vz_smoothed": 0.0},
        ],
    }

    result = _relative_motion_video(video, ego)

    assert result["num_observed_objects_total"] == 1
    assert result["num_repaired_objects_total"] == 1
    assert result["frames"][0]["labels"] == ["car"]
    assert result["frames"][1]["frame_labels"] == ["car"]
    first = result["frames"][0]["objects"][0]
    second = result["frames"][1]["objects"][0]
    assert first["source"] == "observed"
    assert first["is_observed"] is True
    assert second["source"] == "repaired"
    assert second["is_repaired"] is True
    assert second["motion_state"] == "repaired_with_rel_motion"
    assert second["frame_label"] == "car"
    assert abs(second["rel_vx"] - 0.9) < 1e-9


def test_trajectory_motion_evidence_aggregates_frame_level_motion():
    video = {
        "video_id": "vid",
        "frames": [
            _frame(0, [[0, 0, 10, 10]], [0.8], ["car"], [7], [[0, 0, 10]]),
            _frame(1, [[5, 0, 15, 10]], [0.7], ["car"], [7], [[1, 0, 9]]),
        ],
    }
    video["frames"][1]["sources"] = ["tracklet_repair"]
    video["frames"][1]["objects"][0]["source"] = "tracklet_repair"
    video["frames"][1]["objects"][0]["source_type"] = "interpolated_tracklet"
    video["frames"][1]["objects"][0]["position_3d"] = [1, 0, 9]
    ego = {
        "video_id": "vid",
        "frames": [
            {"frame_index": 0, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
            {"frame_index": 1, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
        ],
    }
    relative = _relative_motion_video(video, ego)

    evidence = _trajectory_motion_evidence_video(relative, ego)

    assert evidence["evidence_type"] == "trajectory_motion_evidence"
    assert evidence["num_trajectories"] == 1
    trajectory = evidence["trajectory_motion_evidence"][0]
    assert trajectory["track_id"] == 7
    assert len(trajectory["trajectory_observations"]) == 2
    assert trajectory["trajectory_observations"][0]["bbox"] == [0.0, 0.0, 10.0, 10.0]
    assert trajectory["trajectory_observations"][1]["provenance"]["source"] == "repaired"
    stats = trajectory["trajectory_statistics"]
    assert stats["num_observations"] == 2
    assert stats["frame_span"] == 2
    assert stats["repaired_count"] == 1
    assert stats["observed_count"] == 1
    assert abs(stats["position_z_depth"]["delta"] + 1.0) < 1e-9
    assert abs(stats["rel_speed"]["mean"] - (2 ** 0.5)) < 1e-9
    assert "uncertainty_score" in trajectory["uncertainty"]
    validation = trajectory["causal_motion_fact_validation"]
    assert validation["validation_status"] == "repaired"
    assert validation["repaired"] is True
    assert trajectory["fact_decision_status"] == "Repair"
    assert trajectory["fact_decision"]["symbolic_layer_eligible"] is True


def test_trajectory_motion_validation_detects_id_switch_label_change():
    video = {
        "video_id": "vid",
        "frames": [
            _frame(0, [[0, 0, 10, 10]], [0.9], ["car"], [7], [[0, 0, 10]]),
            _frame(1, [[1, 0, 11, 10]], [0.9], ["truck"], [7], [[0.2, 0, 10]]),
        ],
    }
    ego = {
        "video_id": "vid",
        "frames": [
            {"frame_index": 0, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
            {"frame_index": 1, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
        ],
    }

    evidence = _trajectory_motion_evidence_video(_relative_motion_video(video, ego), ego)

    validation = evidence["trajectory_motion_evidence"][0]["causal_motion_fact_validation"]
    assert validation["validation_status"] == "invalid"
    assert validation["invalid"] is True
    assert "id_switch" in validation["rejection_reasons"]
    trajectory = evidence["trajectory_motion_evidence"][0]
    assert trajectory["fact_decision_status"] == "Discard"
    assert trajectory["fact_decision"]["symbolic_layer_eligible"] is False


def test_motion_significance_marks_static_short_track_low():
    video = {
        "video_id": "vid",
        "frames": [
            _frame(0, [[0, 0, 10, 10]], [0.9], ["car"], [7], [[0, 0, 10]]),
            _frame(1, [[0, 0, 10, 10]], [0.9], ["car"], [7], [[0.01, 0, 10.01]]),
        ],
    }
    ego = {
        "video_id": "vid",
        "frames": [
            {"frame_index": 0, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
            {"frame_index": 1, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
        ],
    }

    trajectory = _trajectory_motion_evidence_video(_relative_motion_video(video, ego), ego)["trajectory_motion_evidence"][0]

    assessment = trajectory["motion_significance_assessment"]
    assert trajectory["motion_significance"] == "low_significance"
    assert assessment["is_low_significance"] is True
    assert {reason["kind"] for reason in assessment["reasons"]} & {"extremely_short_trajectory", "nearly_static", "below_estimated_noise"}
    assert trajectory["fact_decision_status"] == "Keep with uncertainty"
    assert trajectory["fact_decision"]["symbolic_layer_eligible"] is True


def test_motion_significance_marks_stable_motion_high():
    video = {
        "video_id": "vid",
        "frames": [
            _frame(0, [[0, 0, 10, 10]], [0.95], ["car"], [7], [[0, 0, 10]]),
            _frame(1, [[4, 0, 14, 10]], [0.95], ["car"], [7], [[0.4, 0, 9.7]]),
            _frame(2, [[8, 0, 18, 10]], [0.95], ["car"], [7], [[0.8, 0, 9.4]]),
        ],
    }
    ego = {
        "video_id": "vid",
        "frames": [
            {"frame_index": 0, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
            {"frame_index": 1, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
            {"frame_index": 2, "ego_vx_smoothed": 0.0, "ego_vz_smoothed": 0.0},
        ],
    }

    trajectory = _trajectory_motion_evidence_video(_relative_motion_video(video, ego), ego)["trajectory_motion_evidence"][0]

    assert trajectory["causal_motion_fact_validation"]["validation_status"] == "valid"
    assessment = trajectory["motion_significance_assessment"]
    assert trajectory["motion_significance"] == "high_significance"
    assert assessment["is_high_significance"] is True
    assert trajectory["fact_decision_status"] == "Keep"
    assert trajectory["fact_decision"]["symbolic_layer_eligible"] is True
