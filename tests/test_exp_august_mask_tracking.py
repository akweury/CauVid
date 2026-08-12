import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.exp_august import mask_tracking


class ExpAugustMaskTrackingTests(unittest.TestCase):
    def test_hungarian_assignment_uses_mask_class_flow_and_depth_cues(self):
        left = np.zeros((20, 20), dtype=bool)
        left[2:9, 2:9] = True
        right = np.zeros((20, 20), dtype=bool)
        right[10:18, 11:19] = True
        proposals = [
            mask_tracking.MaskProposal(
                track_id=10,
                mask=left,
                bbox=[2, 2, 9, 9],
                label="car",
                confidence=0.9,
                mask_path="left.png",
                flow_warped_mask=left,
                depth=12.0,
            ),
            mask_tracking.MaskProposal(
                track_id=20,
                mask=right,
                bbox=[11, 10, 19, 18],
                label="person",
                confidence=0.8,
                mask_path="right.png",
                flow_warped_mask=right,
                depth=5.0,
            ),
        ]
        detections = [
            {
                "bbox": [11, 10, 19, 18],
                "class": "person",
                "score": 0.8,
                "mask": right,
                "median_depth": 5.2,
            },
            {
                "bbox": [2, 2, 9, 9],
                "class": "car",
                "score": 0.9,
                "mask": left,
                "median_depth": 12.2,
            },
        ]

        matches, unmatched_proposals, unmatched_detections = mask_tracking.associate_proposals(
            proposals,
            detections,
            mask_tracking.HybridMaskTrackingConfig(min_assignment_score=0.5),
        )

        self.assertEqual([(row[0], row[1]) for row in matches], [(0, 1), (1, 0)])
        self.assertEqual(unmatched_proposals, [])
        self.assertEqual(unmatched_detections, [])
        for _proposal, _detection, score, evidence in matches:
            self.assertGreater(score, 0.95)
            self.assertEqual(evidence["mask_support"], "neural_mask")
            self.assertIn("flow_iou", evidence["cues"])
            self.assertIn("depth_consistency", evidence["cues"])

    def test_assignment_rejects_low_scoring_pairs(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:4, 1:4] = True
        proposal = mask_tracking.MaskProposal(
            track_id=1,
            mask=mask,
            bbox=[1, 1, 4, 4],
            label="car",
            confidence=0.9,
            mask_path="mask.png",
        )
        detection = {"bbox": [8, 8, 11, 11], "class": "person", "score": 0.9}

        matches, unmatched_proposals, unmatched_detections = mask_tracking.associate_proposals(
            [proposal],
            [detection],
            mask_tracking.HybridMaskTrackingConfig(min_assignment_score=0.4),
        )

        self.assertEqual(matches, [])
        self.assertEqual(unmatched_proposals, [0])
        self.assertEqual(unmatched_detections, [0])

    def test_auto_backend_records_bytetrack_fallback_without_claiming_masks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            video_root = root / "video-a"
            video_root.mkdir()
            tracks_path = video_root / "tracks.json"
            tracking_state = {
                "videos": ["video-a"],
                "tracks": [
                    {
                        "video_id": "video-a",
                        "frames": [],
                        "output_paths": {"tracks_json": str(tracks_path)},
                    }
                ],
            }
            result = mask_tracking.run(
                detection_state={"detections": [], "detection_args": {"device": "cpu"}},
                tracking_state=tracking_state,
                tracking_args={
                    "output_root": root,
                    "mask_tracking": {
                        "backend": "auto",
                        "sam2_model": str(root / "missing-sam2.pt"),
                    },
                },
                project_root=root,
            )

            self.assertEqual(result["tracking_backend_effective"], "bytetrack")
            metadata = result["tracks"][0]["mask_tracking"]
            self.assertEqual(metadata["status"], "fallback")
            self.assertEqual(metadata["mask_semantics"], "none")
            persisted = json.loads(tracks_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["mask_tracking"]["effective_backend"], "bytetrack")

    def test_explicit_hybrid_backend_fails_when_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "checkpoint not found"):
                mask_tracking.run(
                    detection_state={"detections": [], "detection_args": {"device": "cpu"}},
                    tracking_state={"videos": ["v"], "tracks": []},
                    tracking_args={
                        "output_root": Path(tmp),
                        "mask_tracking": {
                            "backend": "hybrid_mask",
                            "sam2_model": str(Path(tmp) / "missing.pt"),
                        },
                    },
                    project_root=Path(tmp),
                )


if __name__ == "__main__":
    unittest.main()
