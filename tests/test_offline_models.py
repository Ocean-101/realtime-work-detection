"""
Unit Tests for Offline Box Manipulation Dataset & Model Inferences
Validates dataset integrity, YOLO annotations, and offline model inference.
"""

import os
import sys
import json
import unittest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.perception_agent import PerceptionAgent


class TestOfflineDatasetAndModels(unittest.TestCase):

    def test_dataset_structure_and_labels(self):
        """Verify that the dataset generated from clip1.mp4 is properly formatted."""
        dataset_dir = "dataset/box_manipulation_dataset"
        self.assertTrue(os.path.exists(dataset_dir), "Dataset root directory must exist")
        self.assertTrue(os.path.exists(os.path.join(dataset_dir, "data.yaml")), "data.yaml must exist")
        self.assertTrue(os.path.exists(os.path.join(dataset_dir, "action_labels.json")), "action_labels.json must exist")

        train_imgs = os.listdir(os.path.join(dataset_dir, "images", "train"))
        val_imgs = os.listdir(os.path.join(dataset_dir, "images", "val"))
        train_lbls = os.listdir(os.path.join(dataset_dir, "labels", "train"))
        val_lbls = os.listdir(os.path.join(dataset_dir, "labels", "val"))

        self.assertGreater(len(train_imgs), 50, "Should have at least 50 training images")
        self.assertGreater(len(val_imgs), 10, "Should have at least 10 validation images")
        self.assertEqual(len(train_imgs), len(train_lbls), "Every train image must have a label file")
        self.assertEqual(len(val_imgs), len(val_lbls), "Every val image must have a label file")

        # Check normalization of label coordinates
        sample_lbl_path = os.path.join(dataset_dir, "labels", "train", train_lbls[0])
        with open(sample_lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                self.assertIn(cls_id, [0, 1, 2, 3])
                self.assertTrue(0.0 <= cx <= 1.0, f"cx out of bounds: {cx}")
                self.assertTrue(0.0 <= cy <= 1.0, f"cy out of bounds: {cy}")
                self.assertTrue(0.0 <= w <= 1.0, f"w out of bounds: {w}")
                self.assertTrue(0.0 <= h <= 1.0, f"h out of bounds: {h}")

    def test_har_action_model_offline_inference(self):
        """Verify offline PyTorch procedural HAR classifier inference."""
        model_path = "models/har_sequence_classifier.pt"
        self.assertTrue(os.path.exists(model_path), "HAR model weights must exist")

        import torch
        import torch.nn as nn

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", checkpoint)
        self.assertEqual(checkpoint["input_features"], 5)

        # Build inference architecture
        class ProceduralHARClassifier(nn.Module):
            def __init__(self, in_features=5, num_classes=5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )

            def forward(self, x):
                return self.net(x)

        model = ProceduralHARClassifier()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Test S0 (Standby) and S2 (Extracted) feature inputs
        dummy_s0 = torch.tensor([[0.05, 0.0, 0.0, 0.50, 0.25]], dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_s0)
            self.assertEqual(out.shape, (1, 5))

    def test_perception_agent_dual_mode_execution(self):
        """Verify PerceptionAgent executes cleanly with offline model / heuristic fallback."""
        agent = PerceptionAgent()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects, pose, lid_angle = agent.process_frame(dummy_frame)
        self.assertIsInstance(objects, dict)
        self.assertIsInstance(lid_angle, float)

    def test_unified_dataset_and_detector_model(self):
        """Verify unified dataset YAML and newly trained detector_offline.pt."""
        yaml_path = "dataset/unified_detector_dataset/data.yaml"
        self.assertTrue(os.path.exists(yaml_path), "Unified data.yaml must exist")
        model_path = "models/detector_offline.pt"
        self.assertTrue(os.path.exists(model_path), "detector_offline.pt must exist")
        self.assertGreater(os.path.getsize(model_path), 5_000_000, "Model weights file must be > 5MB")


if __name__ == "__main__":
    unittest.main()
