import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.exp_july.perception.trajectory_pattern_runtime_monitor import Step8CRuntimeMonitor


class Step8CRuntimeMonitorTests(unittest.TestCase):
    def test_live_artifacts_metrics_samples_and_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor = Step8CRuntimeMonitor(root, 1, rolling_window=5)
            monitor.handle_batch_event({
                "stage": "batch_stage1", "batch_size": 1, "retry_level": 0,
                "latency": 0.25, "input_count": 1, "valid_count": 0,
                "failed_count": 1, "malformed_count": 1, "token_cost": 12,
                "real_llm_call": True, "backend": "openai_chat_completions",
                "model": "gpt-4.1-mini", "heuristic_fallback": False,
                "timeout_occurred": False, "prompt_tokens": 8,
                "completion_tokens": 4, "total_tokens": 12,
                "token_counts_source": "api_usage", "true_latency_seconds": 0.25,
            })
            monitor.interpretation_complete({
                "track_uid": "video::track_1", "llm_called": True,
                "llm_skipped": False, "cache_hit": False,
                "escalated_to_single": True, "validation_outcome": "validated",
                "latency": 0.25, "gate_reasons": ["ambiguous_pattern_scores"],
            }, {"object_class": "car", "top_candidate_patterns": ["crossing"]})
            monitor.track_complete({
                "video_id": "video", "track_id": 1, "repair_applied": False,
                "resolution_status": "unresolved_uncertain",
                "final_validation_status": "invalid",
                "LLM_preferred_pattern": "crossing", "validated_pattern": "unknown",
                "final_selection_reason": "no_candidate_passed_hard_constraints_original_preserved",
            })
            result = monitor.finalize()

            self.assertEqual(json.loads((root / "runtime_summary.json").read_text())["status"], "completed")
            self.assertTrue((root / "runtime_dashboard.html").exists())
            self.assertIn("Auto-refresh: 3 seconds", (root / "runtime_dashboard.html").read_text())
            self.assertTrue((root / "recent_qualitative_samples.json").exists())
            self.assertTrue((root / "anomaly_alerts.jsonl").read_text().strip())
            with (root / "batch_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["stage"], "batch_stage1")
            self.assertEqual(rows[0]["real_llm_call"], "True")
            self.assertEqual(rows[0]["backend"], "openai_chat_completions")
            self.assertEqual(rows[0]["model"], "gpt-4.1-mini")
            self.assertEqual(rows[0]["total_tokens"], "12")
            self.assertTrue((root / "batch_metrics.jsonl").read_text().strip())
            self.assertEqual(result["completed_units"], 2)
            self.assertEqual(result["counts"]["llm_called"], 1)
            self.assertEqual(result["counts"]["uncertain"], 1)

    def test_no_batches_still_creates_empty_batch_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            monitor = Step8CRuntimeMonitor(root, 0)
            monitor.finalize()
            self.assertEqual(len((root / "batch_metrics.csv").read_text().splitlines()), 1)
            self.assertEqual((root / "batch_metrics.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()
