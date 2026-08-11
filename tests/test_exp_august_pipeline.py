import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from src.exp_august import modules
from src.exp_august import pipeline
from src.exp_august.splits import create_split_manifest


class ExpAugustPipelineTests(unittest.TestCase):
    def test_public_pipeline_has_only_paper_one_stages(self):
        self.assertEqual(len(pipeline.PIPELINE_STEPS), 11)
        self.assertEqual(len(pipeline.STEP_FUNCTION_NAMES), 11)
        self.assertEqual(pipeline.STEP_FUNCTION_NAMES[1], "Object Detection")
        forbidden = ("target", "rule", "causal")
        for name in pipeline.PIPELINE_STEPS:
            self.assertFalse(any(token in name for token in forbidden), name)
        self.assertEqual(pipeline.PIPELINE_STEPS[-1], "symbolic_scene_representation")

    def test_nested_logs_are_suppressed_by_the_public_step_progress(self):
        output = StringIO()
        with patch("sys.stdout", output), patch("sys.stderr", output):
            result = pipeline._tracked(
                pipeline._NullTracker(),
                "04_3d_trajectory_construction",
                lambda: (print("[step 6] positions"), {"videos": ["v1"]})[1],
            )
        self.assertEqual(result["videos"], ["v1"])
        self.assertIn("Step 4 Trajectories", output.getvalue())
        self.assertNotIn("step 6", output.getvalue().lower())
        self.assertLessEqual(len([line for line in output.getvalue().splitlines() if line.strip()]), 4)

    def test_detection_device_summary_is_compact(self):
        with patch("platform.processor", return_value="Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz"):
            summary = pipeline._detection_device_summary({"detection_args": {"device": "cpu"}})
        self.assertEqual(summary, "Device: CPU | Intel Xeon Gold 6338 CPU @ 2.00GHz")
        self.assertLessEqual(len(summary.split(" | ")[1]), 48)

    def test_wandb_tracker_uses_august_full_scale_metadata(self):
        tracker = Mock()
        with patch.dict(
            os.environ,
            {
                "CAUVID_WANDB_ENABLED": "1",
                "CAUVID_WANDB_PROJECT": "project",
                "CAUVID_WANDB_RUN_NAME": "full-seed-726381",
            },
            clear=True,
        ), patch("src.exp_july.wandb_tracking.WandbTracker", return_value=tracker) as tracker_class:
            result = pipeline._wandb_tracker(
                video_ids=None, video_count=961, seed=726381, max_step=11
            )
        self.assertIs(result, tracker)
        kwargs = tracker_class.call_args.kwargs
        self.assertEqual(kwargs["project"], "project")
        self.assertEqual(kwargs["run_name"], "full-seed-726381")
        self.assertEqual(kwargs["config"]["pipeline"], "exp_august")
        self.assertEqual(kwargs["config"]["data_scale"], "full")

    def test_real_video_tqdm_is_forwarded_instead_of_stage_one_of_one(self):
        output = StringIO()
        stream = pipeline._SelectedTqdmStream(output, ((r".*", "Step 4 Trajectories"),))
        stream.write("[step 6] positions_3d:  70%|#######   | 7/10 [00:07<00:03]")
        stream.write("\r[step 6] positions_3d: 100%|##########| 10/10 [00:10<00:00]\n")
        stream.write("another nested bar: 100%|##########| 3/3\n")
        self.assertTrue(stream.saw_progress)
        self.assertIn("10/10", output.getvalue())
        self.assertIn("Step 4 Trajectories: 100%", output.getvalue())
        self.assertNotIn("step 6", output.getvalue().lower())
        self.assertNotIn("3/3", output.getvalue())

    def test_step5_forwards_exactly_three_canonical_mini_step_bars(self):
        output = StringIO()
        stream = pipeline._SelectedTqdmStream(
            output,
            (
                (r"\[step 7\]\s*ego_motion", "Step 5a Ego Motion"),
                (r"\[step 7a\]\s*axis_threshold_segmentation", "Step 5b Axis Threshold Segmentation"),
                (r"\[step 7b\]\s*consensus_merge", "Step 5c Axis Consensus Segmentation"),
            ),
        )
        for description in (
            "[step 7] ego_motion",
            "[step 7a] axis_threshold_segmentation",
            "[step 7a] eval visualizations",
            "[step 7b] consensus_merge",
        ):
            stream.write(f"{description}: 100%|##########| 10/10\n")
        text = output.getvalue()
        self.assertIn("Step 5a Ego Motion: 100%", text)
        self.assertIn("Step 5b Axis Threshold Segmentation: 100%", text)
        self.assertIn("Step 5c Axis Consensus Segmentation: 100%", text)
        self.assertNotIn("eval visualizations", text)

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

    def test_dataset_selection_is_seeded_and_defaults_to_first_generated_seed(self):
        available = ["video_c", "video_a", "video_d", "video_b"]
        expected_manifest = create_split_manifest(available, 2, modules.DATA_SELECTION_SEEDS[0])
        expected = [
            video_id
            for name in ("train", "eval", "test")
            for video_id in expected_manifest[f"{name}_video_ids"]
        ]
        july = Mock()
        july.step1_init.return_value = {"videos": expected}

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"CAUVID_PIPELINE_OUTPUT_PATH": tmp}
        ), patch.object(modules.config, "get_mini_video_ids", return_value=available), patch.object(
            modules, "_july", return_value=july
        ):
            result = modules.dataset_initialization(video_count=2)

        july.step1_init.assert_called_once_with(video_ids=expected, video_count=None)
        self.assertEqual(result["data_selection"]["seed"], modules.DATA_SELECTION_SEEDS[0])
        self.assertEqual(result["data_selection"]["available_seeds"], list(modules.DATA_SELECTION_SEEDS))
        self.assertEqual(result["data_split_manifest"]["counts"], {"train": 2, "eval": 0, "test": 0})

    def test_explicit_video_ids_are_not_reordered(self):
        july = Mock()
        july.step1_init.return_value = {"videos": ["video_b"]}
        july.step7_train_eval_split.side_effect = lambda state: state
        with patch.object(modules, "_july", return_value=july):
            result = modules.dataset_initialization(["video_b", "video_a"], 1, seed=930241)
        july.step1_init.assert_called_once_with(video_ids=["video_b", "video_a"], video_count=1)
        self.assertEqual(result["data_selection"]["method"], "explicit_video_ids")

    def test_step1_creates_august_output_root_on_a_fresh_machine(self):
        july = Mock()
        july.step1_init.return_value = {"videos": ["video-a"]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "missing" / "pipeline_august"
            with patch.dict(
                os.environ, {"CAUVID_PIPELINE_OUTPUT_PATH": str(root)}
            ), patch.object(modules.config, "get_mini_video_ids", return_value=["video-a"]), patch.object(
                modules, "_july", return_value=july
            ):
                modules.dataset_initialization(video_count=1)
            self.assertTrue(root.is_dir())
            self.assertTrue((root / "data_split_manifest.json").is_file())

    def test_native_output_is_isolated_by_scale_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True), patch.object(
            modules.config, "get_output_path", return_value=Path(tmp)
        ):
            debug = modules.get_august_output_root(10, 726381)
            small = modules.get_august_output_root(100, 184957)
            full = modules.get_august_output_root(961, 930241)
        self.assertEqual(debug, Path(tmp) / "pipeline_august" / "debug" / "seed_726381")
        self.assertEqual(small, Path(tmp) / "pipeline_august" / "small" / "seed_184957")
        self.assertEqual(full, Path(tmp) / "pipeline_august" / "full" / "seed_930241")

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
        july.step7a_axis_threshold_segmentation.assert_called_once_with(
            state,
            render_candidate_filter_comparisons=False,
            output_subdir="05b_ego_axis_threshold_segmentation",
            display_step_label="5B",
        )
        july.step7b_optimal_segmentation_selection.assert_called_once_with(
            axis,
            output_subdir="05c_ego_axis_consensus_segmentation",
            display_step_label="5C",
        )

    def test_refinement_diagnostics_are_optional(self):
        state = {
            "videos": ["train", "eval", "test"],
            "step7_train_video_ids": ["train"],
            "step7_eval_video_ids": ["eval"],
            "step7_test_video_ids": ["test"],
        }
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
        with patch.dict(os.environ, {}, clear=True), patch.object(modules, "_july", return_value=july):
            modules.trajectory_refinement(state)
        july.step8h_trajectory_repair_visualization.assert_not_called()
        july.step8i_trajectory_audit_dashboard.assert_not_called()
        july.step8j_trajectory_provenance_audit.assert_not_called()
        clustering_generator = july.step8c_trajectory_clustering.call_args.kwargs["llm_generate"]
        clustering_state = july.step8c_trajectory_clustering.call_args.args[0]
        self.assertTrue(clustering_state["trajectory_refinement_split_policy"]["strict_test_holdout"])
        self.assertEqual(clustering_state["trajectory_refinement_split_policy"]["test_video_ids"], ["test"])
        repair_generator = july.step8d_closed_loop_trajectory_repair.call_args.kwargs["llm_generate"]
        self.assertIs(clustering_generator, modules._offline_refinement_generator)
        self.assertIs(repair_generator, modules._offline_refinement_generator)

    def test_refinement_uses_openai_backend_when_key_is_configured(self):
        state = {"videos": ["v1"]}
        july = Mock()
        for name in (
            "step8_trajectory_repair", "step8a_relative_object_motion", "step8b_signal_evidence",
            "step8c_trajectory_clustering", "step8d_closed_loop_trajectory_repair",
            "step8e_repaired_trajectory_validation", "step8f_trajectory_statistics",
            "step8g_repaired_track_materialization", "step8k_trajectory_handoff",
        ):
            getattr(july, name).return_value = state
        with patch.dict(os.environ, {"OPENAI_API_KEY": "configured"}), patch.object(
            modules, "_july", return_value=july
        ):
            result = modules.trajectory_refinement(state)
        self.assertIsNone(july.step8c_trajectory_clustering.call_args.kwargs["llm_generate"])
        self.assertEqual(result["trajectory_refinement_llm_backend"], "openai")

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
