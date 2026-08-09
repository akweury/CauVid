import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.exp_august import modules
from src.exp_august import pipeline


class ExpAugustPipelineTests(unittest.TestCase):
    def test_public_pipeline_has_only_paper_one_stages(self):
        self.assertEqual(len(pipeline.PIPELINE_STEPS), 11)
        forbidden = ("target", "rule", "causal")
        for name in pipeline.PIPELINE_STEPS:
            self.assertFalse(any(token in name for token in forbidden), name)
        self.assertEqual(pipeline.PIPELINE_STEPS[-1], "symbolic_scene_representation")

    def test_runner_executes_eleven_coherent_modules_in_order(self):
        calls = []

        def stage(name):
            def run(state=None, *_args, **_kwargs):
                calls.append(name)
                current = {"videos": ["v1"]} if state is None or not isinstance(state, dict) else state
                if name == "relative_motion_representation":
                    current = {**current, "relative_object_motion": [{"video_id": "v1"}]}
                return {**current, name: True}

            return run

        patches = {
            name: patch.object(modules, name, side_effect=stage(name))
            for name in pipeline.PIPELINE_STEPS
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patches["dataset_initialization"]:
                with patches["object_detection"], patches["object_tracking"], patches["trajectory_construction_3d"]:
                    with patches["ego_motion_abstraction"], patches["trajectory_refinement"], patches["relative_motion_representation"]:
                        with patches["temporal_video_segmentation"], patches["segment_motion_abstraction"]:
                            with patches["important_object_selection"], patches["symbolic_scene_representation"]:
                                result = pipeline.run_pipeline(output_root=tmp)

        self.assertEqual(calls, list(pipeline.PIPELINE_STEPS))
        self.assertTrue(result["symbolic_scene_representation"])

    def test_max_step_stops_without_reasoning_tail(self):
        first = {"videos": ["v1"]}
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            modules, "dataset_initialization", return_value=first
        ), patch.object(modules, "object_detection") as detection:
            result = pipeline.run_pipeline(max_step=1, output_root=tmp)
        self.assertIs(result, first)
        detection.assert_not_called()

    def test_output_environment_is_scoped_and_restored(self):
        observed = {}

        def initialize(*_args):
            observed["root"] = os.environ.get("CAUVID_PIPELINE_OUTPUT_PATH")
            return {"videos": ["v1"]}

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"CAUVID_PIPELINE_OUTPUT_PATH": "previous"}
        ), patch.object(modules, "dataset_initialization", side_effect=initialize):
            pipeline.run_pipeline(max_step=1, output_root=tmp)
            self.assertEqual(os.environ["CAUVID_PIPELINE_OUTPUT_PATH"], "previous")
        self.assertEqual(Path(observed["root"]), Path(tmp).absolute())

    def test_ego_motion_is_one_active_module(self):
        state = {"videos": ["v1"]}
        axis = {**state, "ego_motion": [{"video_id": "v1"}]}
        selected = {**axis, "final_ego_symbols": [{"video_id": "v1"}]}
        july = Mock()
        july.step7a_axis_threshold_segmentation.return_value = axis
        july.step7b_optimal_segmentation_selection.return_value = selected
        with patch.object(modules, "_july", return_value=july):
            result = modules.ego_motion_abstraction(state)

        self.assertEqual(result["ego_motion_module_status"], "completed")
        self.assertNotIn("step7_status", result)
        july.step7a_axis_threshold_segmentation.assert_called_once()
        july.step7b_optimal_segmentation_selection.assert_called_once_with(axis)

    def test_refinement_diagnostics_are_optional(self):
        state = {"videos": ["v1"]}
        july = Mock()
        for name in (
            "step8_trajectory_repair",
            "step8a_relative_object_motion",
            "step8b_signal_evidence",
            "step8c_trajectory_clustering",
            "step8d_closed_loop_trajectory_repair",
            "step8e_repaired_trajectory_validation",
            "step8f_trajectory_statistics",
            "step8g_repaired_track_materialization",
            "step8k_trajectory_handoff",
        ):
            getattr(july, name).return_value = state
        with patch.object(modules, "_july", return_value=july):
            modules.trajectory_refinement(state)
        july.step8h_trajectory_repair_visualization.assert_not_called()
        july.step8i_trajectory_audit_dashboard.assert_not_called()
        july.step8j_trajectory_provenance_audit.assert_not_called()

    def test_symbolic_output_contains_evaluation_and_traceability_handoffs(self):
        symbolic = [{"video_id": "v1", "num_segments": 2, "num_atoms": 7}]
        state = {
            "videos": ["v1"],
            "important_objects": [{"video_id": "v1", "segments": []}],
            "detection_confidence": 0.8,
            "track_provenance": {"source": "detector"},
        }
        with tempfile.TemporaryDirectory() as tmp, patch(
            "src.exp_driving_videos.modules.logic_atoms_driving_mini.run",
            return_value=symbolic,
        ):
            result = modules.symbolic_scene_representation(state, Path(tmp))
            self.assertEqual(result["symbolic_scene_representation"], symbolic)
            self.assertEqual(result["traceability"]["video_lineage"][0]["symbolic_atoms"], 7)
            self.assertIn("detection_confidence", result["traceability"]["preserved_confidence_fields"])
            self.assertTrue(Path(result["traceability_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
