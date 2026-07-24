import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch


_PERCEPTION_EXPORTS = (
    "step1_init",
    "step2_detection",
    "step3_tracking",
    "step6_positions_3d",
    "step7_ego_motion",
    "step8_trajectory_repair",
    "step8_threshold_epoch_begin",
    "step8a_relative_object_motion",
    "step8b_signal_evidence",
    "step8c_trajectory_pattern_closed_loop",
    "step8d_pattern_refined_validation",
    "step8e_semantic_protection",
    "step8e_visual_semantic_protection",
    "step8f_final_trajectory_validation",
    "step8g_prior_guided_ego_motion_refinement",
    "step8h_visual_relative_motion",
    "step8i_threshold_calibration",
    "step9_temporal_segmentation",
    "step10_segment_object_motion",
)


def _load_pipeline_module():
    perception = types.ModuleType("src.exp_july.perception")

    def unused_step(*_args, **_kwargs):
        raise AssertionError("unexpected perception step")

    for name in _PERCEPTION_EXPORTS:
        setattr(perception, name, unused_step)

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "exp_july"
        / "pipeline.py"
    )
    spec = importlib.util.spec_from_file_location(
        "exp_july_pipeline_wandb_test",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"src.exp_july.perception": perception},
    ):
        spec.loader.exec_module(module)
    return module


pipeline = _load_pipeline_module()


def _fake_wandb(run):
    module = types.ModuleType("wandb")
    module.init = Mock(return_value=run)
    module.Video = Mock()
    module.Artifact = Mock()
    module.Settings = Mock(return_value=object())
    return module


def _fake_run():
    run = Mock()
    run.id = "test-run"
    run.summary = {}
    return run


class ExpJulyWandbTests(unittest.TestCase):
    def test_disabled_does_not_require_wandb_to_be_installed(self):
        state = {"videos": ["video-a"]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "CAUVID_WANDB_ENABLED": "0",
                "CAUVID_PIPELINE_OUTPUT_PATH": tmp,
            },
            clear=True,
        ), patch.dict(sys.modules, {"wandb": None}), patch.object(
            pipeline, "step1_init", return_value=state
        ) as step1:
            result = pipeline.main(max_step=1)

        self.assertIs(result, state)
        step1.assert_called_once_with(video_ids=None, video_count=None)

    def test_wandb_init_failure_is_fail_open(self):
        state = {"videos": ["video-a"]}
        run = _fake_run()
        wandb = _fake_wandb(run)
        wandb.init.side_effect = RuntimeError("init failed")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
            clear=True,
        ), patch.dict(sys.modules, {"wandb": wandb}), patch.object(
            pipeline, "step1_init", return_value=state
        ):
            result = pipeline.main(
                max_step=1,
                wandb_enabled=True,
                wandb_mode="offline",
            )

        self.assertIs(result, state)
        wandb.init.assert_called_once()
        run.finish.assert_not_called()

    def test_enabled_initializes_logs_and_finishes_successfully(self):
        state = {"videos": ["video-a", "video-b"]}
        run = _fake_run()
        wandb = _fake_wandb(run)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
            clear=True,
        ), patch.dict(sys.modules, {"wandb": wandb}), patch.object(
            pipeline, "step1_init", return_value=state
        ):
            result = pipeline.main(
                video_ids=["video-a", "video-b"],
                video_count=2,
                rounds=4,
                max_step=1,
                wandb_enabled=True,
                wandb_project="test-project",
                wandb_run_name="test-name",
                wandb_mode="offline",
            )

        self.assertIs(result, state)
        wandb.init.assert_called_once()
        init_kwargs = wandb.init.call_args.kwargs
        self.assertEqual(init_kwargs["project"], "test-project")
        self.assertEqual(init_kwargs["name"], "test-name")
        self.assertEqual(init_kwargs["mode"], "offline")
        self.assertEqual(init_kwargs["job_type"], "pipeline")
        wandb.Settings.assert_called_once_with(init_timeout=30)
        self.assertIs(init_kwargs["settings"], wandb.Settings.return_value)
        self.assertEqual(
            init_kwargs["config"],
            {
                "pipeline": "exp_july",
                "video_ids": ["video-a", "video-b"],
                "video_count": 2,
                "rounds": 4,
                "max_step": 1,
                "step8_pattern_llm_model": "gpt-4.1-mini",
                "step8c_llm_timeout_seconds": 120.0,
                "step8c_llm_max_attempts": 3,
                "step8c_llm_retry_backoff_seconds": 2.0,
                "step8c_review_interval_tracks": 500,
                "step8_threshold_min_conflicts": 10,
                "step8_threshold_target_quantile": 0.9,
                "step8_threshold_max_relative_change": 0.1,
                "step8_threshold_max_unprotected_flip_rate": 0.02,
                "step8_threshold_validation_fraction": 0.2,
            },
        )
        self.assertEqual(
            Path(init_kwargs["dir"]),
            Path(tmp) / "wandb",
        )

        self.assertEqual(run.log.call_count, 2)
        step_metrics = run.log.call_args_list[0]
        self.assertEqual(step_metrics.kwargs["step"], 1)
        self.assertEqual(step_metrics.args[0]["pipeline/step_name"], "01_init")
        self.assertEqual(step_metrics.args[0]["steps/01_init/video_count"], 2)
        self.assertIn(
            "steps/01_init/duration_seconds",
            step_metrics.args[0],
        )
        final_metrics = run.log.call_args_list[1]
        self.assertEqual(final_metrics.kwargs["step"], 2)
        self.assertEqual(
            final_metrics.args[0]["pipeline/status"],
            "completed",
        )
        self.assertEqual(run.summary["pipeline/status"], "completed")
        self.assertEqual(run.summary["pipeline/steps_logged"], 1)
        run.finish.assert_called_once_with(exit_code=0)

    def test_default_online_tracking_uses_hosted_web_service(self):
        state = {"videos": ["video-a"]}
        run = _fake_run()
        run.url = "https://wandb.ai/test-team/test-project/runs/test-run"
        wandb = _fake_wandb(run)
        observed = {}

        def initialize(**kwargs):
            observed["base_url"] = os.environ.get("WANDB_BASE_URL")
            observed["mode"] = kwargs.get("mode")
            return run

        wandb.init.side_effect = initialize
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {
                "CAUVID_PIPELINE_OUTPUT_PATH": tmp,
                "WANDB_BASE_URL": "http://localhost:8080",
            },
            clear=True,
        ), patch.dict(sys.modules, {"wandb": wandb}), patch.object(
            pipeline, "step1_init", return_value=state
        ):
            result = pipeline.main(max_step=1, wandb_enabled=True)

        self.assertIs(result, state)
        self.assertEqual(observed["mode"], "online")
        self.assertEqual(observed["base_url"], "https://api.wandb.ai")
        self.assertEqual(
            run.summary["pipeline/wandb_base_url"],
            "https://api.wandb.ai",
        )
        self.assertEqual(
            run.summary["pipeline/wandb_web_url"],
            run.url,
        )

    def test_failure_is_finalized_and_original_error_is_reraised(self):
        run = _fake_run()
        wandb = _fake_wandb(run)
        failure = RuntimeError("pipeline failed")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
            clear=True,
        ), patch.dict(sys.modules, {"wandb": wandb}), patch.object(
            pipeline, "step1_init", side_effect=failure
        ):
            with self.assertRaises(RuntimeError) as raised:
                pipeline.main(
                    max_step=1,
                    wandb_enabled=True,
                    wandb_mode="offline",
                )

        self.assertIs(raised.exception, failure)
        final_metrics = run.log.call_args_list[-1]
        self.assertEqual(final_metrics.args[0]["pipeline/status"], "failed")
        self.assertEqual(
            final_metrics.args[0]["pipeline/error_type"],
            "RuntimeError",
        )
        self.assertEqual(
            final_metrics.args[0]["pipeline/error"],
            "pipeline failed",
        )
        self.assertEqual(run.summary["pipeline/status"], "failed")
        self.assertEqual(run.summary["pipeline/steps_logged"], 0)
        self.assertEqual(run.summary["pipeline/stages_attempted"], 1)
        self.assertEqual(run.summary["pipeline/failed_step"], "01_init")
        run.finish.assert_called_once_with(exit_code=1)

    def test_failure_after_a_successful_stage_records_failed_stage(self):
        run = _fake_run()
        wandb = _fake_wandb(run)
        failure = RuntimeError("detection failed")
        initial_state = {
            "videos": ["video-a"],
            "detection_args": {},
        }

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
            clear=True,
        ), patch.dict(sys.modules, {"wandb": wandb}), patch.object(
            pipeline, "step1_init", return_value=initial_state
        ), patch.object(
            pipeline, "step2_detection", side_effect=failure
        ):
            with self.assertRaises(RuntimeError) as raised:
                pipeline.main(
                    max_step=2,
                    wandb_enabled=True,
                    wandb_mode="offline",
                )

        self.assertIs(raised.exception, failure)
        failure_metrics = run.log.call_args_list[1].args[0]
        self.assertEqual(failure_metrics["pipeline/failed_step"], "02_detection")
        self.assertEqual(failure_metrics["pipeline/stage_status"], "failed")
        self.assertIn(
            "steps/02_detection/duration_seconds",
            failure_metrics,
        )
        self.assertEqual(run.summary["pipeline/steps_logged"], 1)
        self.assertEqual(run.summary["pipeline/stages_attempted"], 2)
        self.assertEqual(run.summary["pipeline/failed_step"], "02_detection")

    def test_finish_failure_does_not_mask_pipeline_error(self):
        run = _fake_run()
        run.finish.side_effect = RuntimeError("finish failed")
        wandb = _fake_wandb(run)
        failure = ValueError("original pipeline failure")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
            clear=True,
        ), patch.dict(sys.modules, {"wandb": wandb}), patch.object(
            pipeline, "step1_init", side_effect=failure
        ):
            with self.assertRaises(ValueError) as raised:
                pipeline.main(
                    max_step=1,
                    wandb_enabled=True,
                    wandb_mode="offline",
                )

        self.assertIs(raised.exception, failure)
        run.finish.assert_called_once_with(exit_code=1)

    def test_wandb_log_failure_is_fail_open(self):
        state = {"videos": ["video-a"]}
        run = _fake_run()
        run.log.side_effect = RuntimeError("log failed")
        wandb = _fake_wandb(run)

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
            clear=True,
        ), patch.dict(sys.modules, {"wandb": wandb}), patch.object(
            pipeline, "step1_init", return_value=state
        ):
            result = pipeline.main(
                max_step=1,
                wandb_enabled=True,
                wandb_mode="offline",
            )

        self.assertIs(result, state)
        run.finish.assert_called_once_with(exit_code=0)

    def test_step8h_media_is_capped_by_video_and_audit_files_are_collected(self):
        from src.exp_july.wandb_tracking import WandbTracker

        run = _fake_run()
        wandb = _fake_wandb(run)
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            step_root = output_root / "08h_relative_motion_tracks"
            step_root.mkdir()
            (step_root / "relative_motion_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
            first_video = step_root / "first.mp4"
            second_video = step_root / "second.mp4"
            third_video = step_root / "third.mp4"
            first_video.touch()
            second_video.touch()
            third_video.touch()
            (step_root / "protected_invalid_threshold_conflicts.json").write_text(
                "[]",
                encoding="utf-8",
            )
            review_root = step_root / "epoch_reviews"
            review_root.mkdir()
            (review_root / "epoch_0001_interval_0001_package.json").write_text(
                "{}",
                encoding="utf-8",
            )
            state = {
                "videos": ["video-a", "video-b"],
                "relative_motion_visualizations": [
                    {
                        "video_id": "video-a",
                        "track_id": 1,
                        "visualization_path": str(first_video),
                    },
                    {
                        "video_id": "video-a",
                        "track_id": 2,
                        "visualization_path": str(second_video),
                    },
                    {
                        "video_id": "video-b",
                        "track_id": None,
                        "visualization_path": str(third_video),
                    },
                ],
                "relative_motion_visualization_output_root": step_root,
            }
            with patch.dict(
                os.environ,
                {
                    "CAUVID_PIPELINE_OUTPUT_PATH": tmp,
                    "CAUVID_WANDB_MAX_VIDEOS": "2",
                },
                clear=True,
            ), patch.dict(sys.modules, {"wandb": wandb}):
                tracker = WandbTracker(
                    config={},
                    enabled=True,
                    mode="offline",
                )
                tracker.log_state("08h_important_video_visualization", state)
                tracker.finish(status="completed")

        self.assertEqual(
            wandb.Video.call_args_list,
            [
                call(str(first_video), format="mp4"),
                call(str(third_video), format="mp4"),
            ],
        )
        artifact = wandb.Artifact.return_value
        artifact_names = {
            row.kwargs["name"]
            for row in artifact.add_file.call_args_list
        }
        self.assertEqual(
            artifact_names,
            {
                "08h_relative_motion_tracks/"
                "relative_motion_manifest.json",
                "08h_relative_motion_tracks/"
                "protected_invalid_threshold_conflicts.json",
                "08h_relative_motion_tracks/epoch_reviews/"
                "epoch_0001_interval_0001_package.json",
            },
        )
        run.log_artifact.assert_called_once_with(artifact)

    def test_each_disk_manifest_is_attributed_to_its_own_stage_once(self):
        from src.exp_july.wandb_tracking import WandbTracker

        run = _fake_run()
        wandb = _fake_wandb(run)
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            detection_root = output_root / "02_detection"
            tracking_root = output_root / "03_tracking"
            detection_root.mkdir()
            tracking_root.mkdir()
            (detection_root / "detection_manifest.json").write_text(
                '{"num_accepted_detections": 3}',
                encoding="utf-8",
            )
            (tracking_root / "tracks_manifest.json").write_text(
                '{"num_tracks": 4}',
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"CAUVID_PIPELINE_OUTPUT_PATH": tmp},
                clear=True,
            ), patch.dict(sys.modules, {"wandb": wandb}):
                tracker = WandbTracker(
                    config={},
                    enabled=True,
                    mode="offline",
                )
                tracker.log_state(
                    "02_detection",
                    {
                        "videos": ["video-a"],
                        "detections": [1, 2, 3],
                        "detection_output_root": detection_root,
                    },
                )
                tracker.log_state(
                    "03_tracking",
                    {
                        "videos": ["video-a"],
                        "detections": [1, 2, 3],
                        "tracks": [1, 2, 3, 4],
                        "detection_output_root": detection_root,
                        "tracking_output_root": tracking_root,
                    },
                )
                tracker.finish(status="completed")

        detection_metrics = run.log.call_args_list[0].args[0]
        tracking_metrics = run.log.call_args_list[1].args[0]
        self.assertEqual(
            detection_metrics[
                "steps/02_detection/manifests/"
                "detection_manifest/num_accepted_detections"
            ],
            3.0,
        )
        self.assertEqual(
            tracking_metrics[
                "steps/03_tracking/manifests/tracks_manifest/num_tracks"
            ],
            4.0,
        )
        self.assertFalse(
            any(
                "detection_manifest" in key
                for key in tracking_metrics
            )
        )

    def test_cli_parses_wandb_options(self):
        argv = [
            "pipeline.py",
            "--wandb",
            "--wandb-project",
            "project-a",
            "--wandb-run-name",
            "run-a",
            "--wandb-mode",
            "offline",
        ]
        with patch.object(sys, "argv", argv):
            args = pipeline._parse_args()

        self.assertTrue(args.wandb_enabled)
        self.assertEqual(args.wandb_project, "project-a")
        self.assertEqual(args.wandb_run_name, "run-a")
        self.assertEqual(args.wandb_mode, "offline")

        with patch.object(sys, "argv", ["pipeline.py", "--no-wandb"]):
            args = pipeline._parse_args()
        self.assertFalse(args.wandb_enabled)


if __name__ == "__main__":
    unittest.main()
