"""
Unit Tests for Offline Model Weights, Checkpoints & Standalone Inferences
ISRO Smart India Hackathon (SIH) | Problem Statement ID: 26174
Validates zero-cloud local neural execution across YOLOv8, PyTorch HAR, and 3D Pose.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from ultralytics import YOLO
from src.agents.perception_agent import PerceptionAgent


class TestOfflineModels(unittest.TestCase):

    def test_detector_offline_weights_and_classes(self):
        """Verify custom-trained YOLOv8 weights exist and have expected experiment classes."""
        model_path = os.path.join(WORKSPACE_ROOT, "models", "detector_offline.pt")
        self.assertTrue(os.path.exists(model_path), f"Offline detector missing: {model_path}")
        self.assertGreater(os.path.getsize(model_path), 5_000_000, "Model checkpoint must be > 5 MB")

        # Load weights purely offline
        model = YOLO(model_path)
        self.assertEqual(model.task, "detect")
        names = model.names
        self.assertEqual(len(names), 6, f"Expected 6 classes, got {len(names)}: {names}")

        expected_classes = ["container_box", "container_lid", "red_box", "yellow_box", "operator_hand", "human_body"]
        for expected in expected_classes:
            self.assertIn(expected, names.values(), f"Expected class {expected} not found in model names")

        # Run offline inference on a synthetic test image
        dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
        results = model(dummy_img, verbose=False, conf=0.25)
        self.assertIsNotNone(results)
        self.assertEqual(len(results), 1)

    def test_har_action_model_offline_inference(self):
        """Verify offline PyTorch procedural HAR classifier inference."""
        model_path = os.path.join(WORKSPACE_ROOT, "models", "har_sequence_classifier.pt")
        self.assertTrue(os.path.exists(model_path), f"HAR model weights missing: {model_path}")

        # Model Architecture definition (matches trained checkpoint)
        class ProceduralHARClassifier(nn.Module):
            def __init__(self, in_features=8, num_classes=5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )

            def forward(self, x):
                return self.net(x)

        model = ProceduralHARClassifier(in_features=8, num_classes=5)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        model.eval()

        # Test dummy kinematic feature vector with 8 features
        dummy_input = torch.tensor([[0.20, 45.0, 1.0, 0.15, 80.0, 110.0, 0.1, 0.2]], dtype=torch.float32)
        with torch.no_grad():
            output = model(dummy_input)
            self.assertEqual(output.shape, (1, 5))
            predicted_stage = torch.argmax(output, dim=1).item()
            self.assertIn(predicted_stage, range(5))

    def test_har_telemetry_classifier_offline_inference(self):
        """Verify 8-dimensional telemetry HAR classifier inference."""
        model_path = os.path.join(WORKSPACE_ROOT, "models", "har_telemetry_classifier.pt")
        self.assertTrue(os.path.exists(model_path), f"HAR telemetry model missing: {model_path}")

        class ProceduralTelemetryHARClassifier(nn.Module):
            def __init__(self, in_features=8, num_classes=5):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Dropout(0.10),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )

            def forward(self, x):
                return self.net(x)

        model = ProceduralTelemetryHARClassifier()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        model.eval()

        dummy_telemetry = torch.tensor([[0.5, 45.0, 1.0, 0.12, 85.0, 120.0, 0.10, 0.25]], dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_telemetry)
            self.assertEqual(out.shape, (1, 5))

    def test_perception_agent_standalone_offline_execution(self):
        """Verify PerceptionAgent executes cleanly with offline model and generates 3D pose."""
        agent = PerceptionAgent(
            model_path="models/detector_offline.pt",
            pose_model_path="models/yolov8n-pose.pt"
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        objects, pose, lid_angle = agent.process_frame(dummy_frame)

        self.assertIsInstance(objects, dict)
        self.assertIsInstance(lid_angle, float)
        self.assertIsNotNone(pose)
        self.assertTrue(hasattr(pose, "joints"))


if __name__ == "__main__":
    unittest.main()
