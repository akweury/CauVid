import unittest
from types import SimpleNamespace

import numpy as np

from src.exp_august.contracts import (
    DepthUnit,
    HypothesisBeam,
    MotionState,
    Observability,
    Step5ConfigSnapshot,
)
from src.exp_august.contracts.codec import hash_payload
from src.exp_august.inference.step05_joint_world_reconstruction import (
    _hypothesis,
    _motion_alternative_hypotheses,
    _rank_hypotheses,
)


def _point(x, y, z, mad=0.01):
    return SimpleNamespace(
        median=SimpleNamespace(x=x, y=y, z=z),
        mad=SimpleNamespace(x=mad, y=mad, z=mad),
    )


def _observation(track_id, frame, z):
    return SimpleNamespace(
        observation_id=f"geometry:{track_id}:{frame}",
        track_id=track_id,
        frame_index=frame,
        timestamp_s=float(frame),
        scale_id="scale:relative",
        validation_passed=True,
        points=_point(2.0, 0.5, z),
    )


def _track(track_id, class_name, depths):
    return SimpleNamespace(
        track_id=track_id,
        primary_class=class_name,
        observations=tuple(
            _observation(track_id, frame, depth)
            for frame, depth in enumerate(depths)
        ),
        unavailable_observations=(),
    )


def _pose(source, target):
    return SimpleNamespace(
        pose_id=f"pose:{source}:{target}",
        source_frame_index=source,
        target_frame_index=target,
        source_timestamp_s=float(source),
        target_timestamp_s=float(target),
        rotation_source_to_target=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        translation_direction_source_to_target=SimpleNamespace(x=0.0, y=0.0, z=-1.0),
        inlier_fraction=0.95,
        median_epipolar_residual_px=0.05,
    )


class ExpAugustStep05WorldReconstructionTests(unittest.TestCase):
    def test_reconstructs_ego_and_separates_static_from_moving_objects(self):
        static_track = _track("track:sign", "traffic sign", (10.0, 9.0, 8.0))
        moving_track = _track("track:car", "car", (10.0, 10.0, 10.0))
        manifest = SimpleNamespace(
            video_id="synthetic",
            canonical_fps=1.0,
            camera_motion=SimpleNamespace(poses=(_pose(0, 1), _pose(1, 2))),
            tracks=(static_track, moving_track),
        )
        scale = SimpleNamespace(
            scale_id="scale:relative",
            observability=Observability.RELATIVE,
            limitations=("no_supported_meters_conversion",),
        )
        config = Step5ConfigSnapshot(
            top_k=5,
            minimum_motion_observations=2,
            static_displacement_threshold=0.25,
            moving_displacement_threshold=0.75,
            static_scale_residual_threshold=0.50,
            fallback_scale_residual_threshold=0.15,
            uncertainty_sigma_multiplier=2.0,
        )
        hypothesis, counts = _hypothesis(
            manifest=manifest,
            scale=scale,
            config=config,
            config_sha256=hash_payload(config),
            source_geometry_sha256="0" * 64,
        )

        self.assertEqual(counts, (2, 1, 6, 6))
        self.assertEqual(hypothesis.coordinate_unit, DepthUnit.RELATIVE_UNIT)
        self.assertFalse(hypothesis.metric_scale_claimed)
        self.assertEqual(hypothesis.world_frame_status, "global")
        centers = [
            (row.position.x, row.position.y, row.position.z)
            for row in hypothesis.ego_components[0].poses
        ]
        self.assertTrue(np.allclose(centers, ((0, 0, 0), (0, 0, 1), (0, 0, 2))))
        states = {row.track_id: row.motion_state for row in hypothesis.object_trajectories}
        self.assertEqual(states["track:sign"], MotionState.STATIC)
        self.assertEqual(states["track:car"], MotionState.MOVING)
        self.assertFalse(hypothesis.unresolved_object_observation_ids)

        beam = HypothesisBeam(
            beam_id="beam0:synthetic",
            top_k=5,
            hypotheses=(hypothesis,),
        )
        self.assertEqual(beam.iteration, 0)
        self.assertEqual(beam.hypotheses[0].rank, 1)

    def test_ambiguous_motion_creates_bounded_static_and_moving_alternatives(self):
        ambiguous_track = _track("track:maybe", "car", (10.0, 9.3, 8.6))
        static_anchor = _track("track:sign", "traffic sign", (10.0, 9.0, 8.0))
        manifest = SimpleNamespace(
            video_id="synthetic-ambiguous",
            canonical_fps=1.0,
            camera_motion=SimpleNamespace(poses=(_pose(0, 1), _pose(1, 2))),
            tracks=(ambiguous_track, static_anchor),
        )
        scale = SimpleNamespace(
            scale_id="scale:relative",
            observability=Observability.RELATIVE,
            limitations=("no_supported_meters_conversion",),
        )
        config = Step5ConfigSnapshot(
            top_k=3,
            minimum_motion_observations=2,
            static_displacement_threshold=0.25,
            moving_displacement_threshold=0.75,
            static_scale_residual_threshold=0.50,
            fallback_scale_residual_threshold=0.15,
            uncertainty_sigma_multiplier=2.0,
        )
        parent, _ = _hypothesis(
            manifest=manifest,
            scale=scale,
            config=config,
            config_sha256=hash_payload(config),
            source_geometry_sha256="1" * 64,
        )
        parent_state = {
            row.track_id: row.motion_state for row in parent.object_trajectories
        }
        self.assertEqual(parent_state["track:maybe"], MotionState.AMBIGUOUS)
        alternatives = _motion_alternative_hypotheses(parent)
        self.assertEqual(len(alternatives), 2)
        alternative_states = {
            next(
                row.motion_state
                for row in hypothesis.object_trajectories
                if row.track_id == "track:maybe"
            )
            for hypothesis in alternatives
        }
        self.assertEqual(alternative_states, {MotionState.STATIC, MotionState.MOVING})
        ranked = _rank_hypotheses([parent, *alternatives], top_k=3)
        self.assertEqual(ranked[0].hypothesis_id, parent.hypothesis_id)
        self.assertEqual(tuple(row.rank for row in ranked), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
