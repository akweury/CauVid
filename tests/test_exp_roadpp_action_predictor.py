import io
import unittest
from contextlib import redirect_stdout

from src.exp_roadpp.step03_action_predictor import ActionPredictor
from src.exp_roadpp.step03_rule_learner import _to_extended_clauses
from src.exp_roadpp.utils_data import build_signature


class ActionPredictorTest(unittest.TestCase):
    def test_classify_transition_detects_unchanged_added_and_removed(self) -> None:
        unchanged = build_signature("action", "walking", 0, "pedestrian", "pedestrian", "tube-1")
        removed = build_signature("location", "road", 0, "car", "car", "tube-2")
        added = build_signature("action", "stopped", 1, "pedestrian", "pedestrian", "tube-1")
        labels = {
            "action_labels": ["walking", "stopped"],
            "av_action_labels": [],
            "loc_labels": ["road"],
        }
        model = ActionPredictor([], labels, "cpu")

        transition = model.classify_transition(
            [unchanged, removed],
            [unchanged, added],
        )

        self.assertEqual(transition["unchanged"], [unchanged])
        self.assertEqual(transition["added"], [added])
        self.assertEqual(transition["removed"], [removed])

    def test_classify_transition_does_not_classify_duplicate_facts_as_new(self) -> None:
        repeated = build_signature("action", "walking", 0, "pedestrian", "pedestrian", "tube-1")
        labels = {
            "action_labels": ["walking"],
            "av_action_labels": [],
            "loc_labels": [],
        }
        model = ActionPredictor([], labels, "cpu")

        transition = model.classify_transition([repeated, repeated], [repeated])

        self.assertEqual(transition["unchanged"], [repeated])
        self.assertEqual(transition["added"], [])
        self.assertEqual(transition["removed"], [])

        transition = model.classify_transition([repeated], [repeated, repeated])
        self.assertEqual(transition["unchanged"], [repeated, repeated])
        self.assertEqual(transition["added"], [])
        self.assertEqual(transition["removed"], [])

    def test_extended_clause_conversion_preserves_body_scopes(self) -> None:
        body = build_signature("action", "walking", 0, "pedestrian", "pedestrian", None)
        extra_body = build_signature("location", "road", 0, "car", "car", None)
        head = build_signature("action", "stopped", 1, "pedestrian", "pedestrian", None)
        metrics = {
            "support": 10,
            "coverage": 0.5,
            "confidence": 0.9,
            "recall": 0.5,
            "confidence_gain": 0.1,
            "f1": 0.6,
        }
        clauses = _to_extended_clauses(
            [{
                "group_id": 0,
                "head": (0, head[:-2]),
                "extra_atoms": [(0, "scene", extra_body[:-2])],
                "metrics": metrics,
            }],
            {"body_groups": {body[:-2]: {}}},
        )

        self.assertEqual(clauses[0]["body_scopes"], ["entity", "scene"])

    def test_predict_requires_all_extended_clause_bodies(self) -> None:
        body = build_signature("action", "walking", 0, "pedestrian", "pedestrian", "tube-1")
        context = build_signature("location", "road", 0, "car", "car", "tube-2")
        head = build_signature("action", "stopped", 1, "pedestrian", "pedestrian", None)
        labels = {
            "action_labels": ["walking", "stopped"],
            "av_action_labels": [],
            "loc_labels": ["road"],
        }
        model = ActionPredictor(
            [{
                "bodies": [body[:-2], context[:-2]],
                "body_scopes": ["entity", "scene"],
                "head": head[:-2],
                "confidence": 0.9,
            }],
            labels,
            "cpu",
        )

        self.assertEqual(model.predict([body]), [body])
        predicted = model.predict([body, context])
        expected_head = build_signature(
            "action", "stopped", 1, "pedestrian", "pedestrian", "tube-1"
        )
        self.assertEqual(predicted, [expected_head, context])
        rule_metrics = model.eval(
            [body, context], predicted, [expected_head, context]
        )["rule_matches"]
        self.assertEqual(rule_metrics["fired_rule_count"], 1)
        self.assertEqual(rule_metrics["correct_fired_rule_count"], 1)
        self.assertEqual(rule_metrics["incorrect_fired_rule_count"], 0)
        self.assertEqual(rule_metrics["transition_count"], 1)
        self.assertEqual(rule_metrics["predicted_transition_count"], 1)
        self.assertEqual(rule_metrics["unpredicted_transition_count"], 0)
        self.assertEqual(rule_metrics["transition_recall"], 1.0)

        missing_context_eval = model.eval([body], [body], [expected_head])
        self.assertEqual(
            missing_context_eval["rule_matches"]["fired_rule_count"],
            0,
        )
        self.assertEqual(
            missing_context_eval["rule_matches"]["unpredicted_transition_count"],
            1,
        )

        entity_scoped_model = ActionPredictor(
            [{
                "bodies": [body[:-2], context[:-2]],
                "body_scopes": ["entity", "entity"],
                "head": head[:-2],
                "confidence": 0.9,
            }],
            labels,
            "cpu",
        )
        self.assertEqual(entity_scoped_model.predict([body, context]), [body, context])

        same_entity_context = build_signature(
            "location", "road", 0, "car", "car", "tube-1"
        )
        entity_prediction = entity_scoped_model.predict([body, same_entity_context])
        self.assertEqual(entity_prediction, [expected_head, same_entity_context])
        entity_rule_matches = entity_scoped_model.eval(
            [body, same_entity_context],
            entity_prediction,
            [expected_head, same_entity_context],
        )["rule_matches"]
        self.assertEqual(entity_rule_matches["correct_fired_rule_count"], 1)
        self.assertEqual(
            entity_scoped_model.eval(
                [body, context], [body, context], [expected_head, context]
            )["rule_matches"]["unpredicted_transition_count"],
            1,
        )

    def test_rule_evaluation_counts_incorrect_fired_heads(self) -> None:
        body = build_signature("action", "walking", 0, "pedestrian", "pedestrian", "tube-1")
        context = build_signature("location", "road", 0, "car", "car", "tube-2")
        head = build_signature("action", "stopped", 1, "pedestrian", "pedestrian", None)
        labels = {
            "action_labels": ["walking", "stopped"],
            "av_action_labels": [],
            "loc_labels": ["road"],
        }
        model = ActionPredictor(
            [{
                "bodies": [body[:-2], context[:-2]],
                "body_scopes": ["entity", "scene"],
                "head": head[:-2],
                "confidence": 0.9,
            }],
            labels,
            "cpu",
        )

        predicted = model.predict([body, context])
        metrics = model.eval(
            [body, context], predicted, [body, context]
        )["rule_matches"]

        self.assertEqual(metrics["fired_rule_count"], 1)
        self.assertEqual(metrics["correct_fired_rule_count"], 0)
        self.assertEqual(metrics["incorrect_fired_rule_count"], 1)
        self.assertEqual(metrics["transition_count"], 0)

    def test_false_positive_logs_rule_ground_truth_and_body_truth(self) -> None:
        body = build_signature("action", "walking", 0, "pedestrian", "pedestrian", "tube-1")
        context = build_signature("location", "road", 0, "car", "car", "tube-2")
        head = build_signature("action", "stopped", 1, "pedestrian", "pedestrian", None)
        labels = {
            "action_labels": ["walking", "stopped"],
            "av_action_labels": [],
            "loc_labels": ["road"],
        }
        model = ActionPredictor(
            [{
                "bodies": [body[:-2], context[:-2]],
                "body_scopes": ["entity", "scene"],
                "head": head[:-2],
                "confidence": 0.9,
            }],
            labels,
            "cpu",
        )

        predicted = model.predict([body, context])
        log = io.StringIO()
        with redirect_stdout(log):
            model.eval([body, context], predicted, [body])

        output = log.getvalue()
        self.assertIn("[Event mismatch: added]", output)
        self.assertIn("[Event mismatch: removed]", output)
        self.assertIn(
            "Ground-truth transition for this exact fact: unchanged "
            "(in_current=True, in_ground_truth_next=True)",
            output,
        )
        self.assertIn("Fired rule #0", output)
        self.assertIn("Current-frame body atom #0:", output)
        self.assertIn("Current-frame body atom #1:", output)
        self.assertIn("tube_uid', 'tube-1'", output)
        self.assertIn("tube_uid', 'tube-2'", output)
        self.assertIn("[Ground-truth atom]", output)
        self.assertNotIn("[Ground-truth added]", output)
        self.assertNotIn("[Ground-truth removed]", output)
        self.assertNotIn("tube_uid', 'tube-2'", output)
        self.assertIn("Full body true in ground truth=False", output)


if __name__ == "__main__":
    unittest.main()
