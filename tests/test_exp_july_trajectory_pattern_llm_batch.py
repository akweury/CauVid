import tempfile
import unittest
from pathlib import Path

from src.exp_july.perception.trajectory_pattern_llm_batch import PATTERNS, RESIDUALS, process_tracks


def item(track_id=1, confidence=.95, state="valid", issues=None, ambiguous=False):
    candidates=[]
    for index,pattern in enumerate(PATTERNS):
        value=1.0 if ambiguous else (0.0 if pattern=="stationary" else 10.0+index)
        candidates.append({"pattern_id":pattern,"residual_vector":{name:value for name in RESIDUALS}})
    track={"video_id":"video","track_id":track_id,"object_class":"car","direction":"stationary",
      "persistence":.9,"confidence":confidence,"bbox_size":{},"position":{},"relative_motion":{},
      "provenance":{"source":"observed"},"source_validation":{"validation_status":state,"rejection_reasons":issues or []},
      "source_decision":"Keep" if state=="valid" else "Discard"}
    return {"track":track,"candidates":candidates}


def stage1(rows,confidence=.9,planning=True):
    return {"results":[{"track_uid":row["track_uid"],"assessments":[{
      "pattern_id":pattern,"plausibility":.8 if pattern=="approaching" else .2,
      "ignorable_errors":[],"structural_conflicts":[],"explanation":"independent"} for pattern in PATTERNS],
      "requires_repair_planning":planning,"batch_confidence":confidence,"batch_conflicts":[]} for row in rows]}


def stage2(rows):
    return {"results":[{"track_uid":row["track_uid"],"repair_recommendations":{
      pattern:["outlier_removal"] for pattern in PATTERNS}} for row in rows]}


class Step8CBatchTests(unittest.TestCase):
    def test_clear_track_bypasses_llm(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            results,telemetry,_=process_tracks([item()],Path(tmp),lambda kind,prompt:calls.append(kind),10)
        row=telemetry["video::track_1"]
        self.assertFalse(row["llm_called"])
        self.assertTrue(row["llm_skipped"])
        self.assertEqual(row["validation_outcome"],"deterministic_bypass")
        self.assertFalse(calls)
        self.assertEqual(len(results["video::track_1"]),len(PATTERNS))

    def test_missing_id_retries_only_failed_track_then_runs_stage_two(self):
        calls=[]
        def invoke(kind,prompt):
            import json
            rows=json.loads(prompt.split("inputs=",1)[1]);calls.append((kind,[row["track_uid"] for row in rows]))
            if kind=="batch_stage1" and len(rows)>1:return stage1(rows[:1])
            if "stage1" in kind:return stage1(rows)
            return stage2(rows)
        items=[item(1,.5,"invalid",["depth_jump"],True),item(2,.5,"invalid",["depth_jump"],True)]
        with tempfile.TemporaryDirectory() as tmp:
            results,telemetry,_=process_tracks(items,Path(tmp),invoke,10)
        stage1_calls=[uids for kind,uids in calls if kind=="batch_stage1"]
        self.assertEqual(stage1_calls[0],["video::track_1","video::track_2"])
        self.assertEqual(stage1_calls[1],["video::track_2"])
        self.assertEqual(telemetry["video::track_1"]["retry_count"],0)
        self.assertEqual(telemetry["video::track_2"]["retry_count"],1)
        self.assertIn("outlier_removal",results["video::track_2"][0]["recommended_repairs"])

    def test_equivalent_track_reuses_signature_cache(self):
        calls=[]
        def invoke(kind,prompt):
            import json
            rows=json.loads(prompt.split("inputs=",1)[1]);calls.append(kind)
            return stage1(rows,planning=False) if "stage1" in kind else stage2(rows)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            process_tracks([item(1,.5,"invalid",["conflict"],True)],root,invoke,10)
            _,telemetry,_=process_tracks([item(99,.5,"invalid",["conflict"],True)],root,invoke,10)
        row=telemetry["video::track_99"]
        self.assertTrue(row["cache_hit"])
        self.assertTrue(row["llm_skipped"])
        self.assertEqual(row["validation_outcome"],"signature_cache_hit")


if __name__=="__main__":unittest.main()