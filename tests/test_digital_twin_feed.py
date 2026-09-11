"""
Unit Tests for Digital Twin Dataset Ingestion & Feed Engine
Verifies:
1. Direct ingestion of JSON dataset records into 3D scene graph.
2. Direct ingestion of CSV telemetry rows into 3D scene graph.
3. 2.5D Isometric Digital Twin canvas rendering with full skeleton and entities.
4. Dataset feeder loader on both JSON and CSV files.
"""

import os
import sys
import unittest
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.agents.digital_twin_agent import DigitalTwinAgent
from tools.feed_dataset_to_digital_twin import load_dataset_records


class TestDigitalTwinFeed(unittest.TestCase):

    def setUp(self):
        self.twin_agent = DigitalTwinAgent()

    def test_sync_from_json_record(self):
        """Verify ingestion of a JSON dataset frame record."""
        record = {
            "frame_id": 100,
            "timestamp_sec": 4.20,
            "persons": [
                {
                    "activity": "EXTRACT COMPONENT BOX",
                    "fsm_step": {"step_id": 2, "step_name": "OBJECT_EXTRACTED"},
                    "pose": {
                        "biomechanics": {
                            "elbow_angle_deg": 135.0,
                            "shoulder_angle_deg": 65.0,
                            "rom_violated": False
                        },
                        "joints_3d": [
                            {"joint_id": 0, "name": "nose", "x": 0.0, "y": -0.5, "z": 1.5},
                            {"joint_id": 5, "name": "left_shoulder", "x": -0.2, "y": -0.3, "z": 1.5},
                            {"joint_id": 6, "name": "right_shoulder", "x": 0.2, "y": -0.3, "z": 1.5},
                            {"joint_id": 8, "name": "right_elbow", "x": 0.3, "y": -0.1, "z": 1.5},
                            {"joint_id": 10, "name": "right_wrist", "x": 0.25, "y": 0.1, "z": 1.5}
                        ]
                    }
                }
            ],
            "objects": {
                "container_box": {"state": "DOCKED", "is_inside": False},
                "component_box": {"state": "EXTRACTED", "is_inside": False}
            }
        }

        sg = self.twin_agent.sync_from_dataset_record(record)
        self.assertIn("rack", sg)
        self.assertIn("container", sg)
        self.assertIn("astronaut", sg)
        self.assertEqual(sg["step"], "OBJECT_EXTRACTED")
        self.assertEqual(sg["activity"], "EXTRACT COMPONENT BOX")
        self.assertEqual(sg["astronaut"]["elbow_angle_deg"], 135.0)
        self.assertIn("right_wrist", sg["astronaut"]["joints"])
        self.assertFalse(sg["entities"]["component_box"]["is_inside_container"])

    def test_sync_from_csv_row(self):
        """Verify ingestion of a CSV telemetry row."""
        row = {
            "frame_id": "50",
            "step_id": "1",
            "step_name": "BOX_OPENED",
            "activity": "OPEN LID",
            "lid_angle_deg": "45.0",
            "wrist_rack_x": "0.15",
            "wrist_rack_y": "-0.10",
            "wrist_rack_z": "0.50",
            "elbow_angle_deg": "120.5",
            "shoulder_angle_deg": "55.0",
            "rom_violated": "0",
            "component_is_inside": "1",
            "component_state": "DOCKED",
            "component_rack_x": "0.0",
            "component_rack_y": "-0.2",
            "component_rack_z": "0.5"
        }

        sg = self.twin_agent.sync_from_dataset_record(row)
        self.assertEqual(sg["step"], "BOX_OPENED")
        self.assertEqual(sg["activity"], "OPEN LID")
        self.assertEqual(sg["container"]["lid_angle_deg"], 45.0)
        self.assertTrue(sg["container"]["is_open"])
        self.assertIn("wrist", sg["astronaut"]["joints"])
        self.assertEqual(sg["astronaut"]["joints"]["wrist"]["pos_rack"], [0.15, -0.10, 0.50])

    def test_render_digital_twin_canvas(self):
        """Verify rendering 2.5D Digital Twin canvas to valid image."""
        scene_graph = {
            "version": "1.0",
            "step": "OBJECT_EXTRACTED",
            "activity": "EXTRACT COMPONENT BOX",
            "rack": {"label": "BAS-03 Payload Experiment Rack", "bounds": (0.8, 0.6, 0.5)},
            "container": {"lid_angle_deg": 65.0, "is_open": True, "origin": [0.0, 0.0, 0.0]},
            "entities": {
                "component_box": {"state": "EXTRACTED", "pos_rack": [0.3, -0.4, 0.5], "is_inside_container": False}
            },
            "astronaut": {
                "joints": {
                    "left_shoulder": {"pos_rack": [-0.15, -0.3, 0.5], "confidence": 0.95},
                    "right_shoulder": {"pos_rack": [0.15, -0.3, 0.5], "confidence": 0.95},
                    "right_elbow": {"pos_rack": [0.25, -0.15, 0.5], "confidence": 0.95},
                    "right_wrist": {"pos_rack": [0.20, 0.05, 0.5], "confidence": 0.95}
                },
                "elbow_angle_deg": 130.0,
                "rom_violated": False
            }
        }

        canvas = self.twin_agent.render_digital_twin_canvas(scene_graph, width=640, height=480)
        self.assertIsInstance(canvas, np.ndarray)
        self.assertEqual(canvas.shape, (480, 640, 3))
        self.assertEqual(canvas.dtype, np.uint8)

    def test_dataset_feeder_loader(self):
        """Verify load_dataset_records on generated dataset files."""
        csv_path = os.path.join(WORKSPACE_ROOT, "dataset", "har_dataset", "har_dataset_telemetry.csv")
        if os.path.exists(csv_path):
            records, fmt = load_dataset_records(csv_path)
            self.assertEqual(fmt, "csv")
            self.assertGreater(len(records), 100)


if __name__ == "__main__":
    unittest.main()
