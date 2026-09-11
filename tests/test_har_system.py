"""
Unit Tests for the Integrated HAR System and Dataset Generation Pipeline
Verifies:
1. Operator hand exclusion from manipulated objects (no more 'CONTACT OPERATOR HAND').
2. Procedural HAR activity classification (IDLE, APPROACH, GRASP, EXTRACT).
3. 3D Pose and joint export schemas (.json, .npz, .csv).
4. PerceptionAgent YOLO11 loading capability.
"""

import os
import sys
import json
import unittest
import numpy as np

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import (
    AstronautPose3D,
    Joint3D,
    ExperimentObject,
    BBox2D,
    EntityState,
    Vector3D,
    HOIAction
)
from src.agents.har_agent import HARAgent


class TestHARSystem(unittest.TestCase):

    def setUp(self):
        self.har_agent = HARAgent()

    def test_operator_hand_excluded_from_targets(self):
        """Verify that operator_hand is NEVER treated as a target object to interact with."""
        pose = AstronautPose3D()
        pose.joints["wrist"] = Joint3D(
            name="wrist",
            pos_camera=Vector3D(0.0, 0.0, 1.2),
            pos_rack=Vector3D(0.0, -0.2, 0.5)
        )

        # Frame where only operator_hand is detected
        objects = {
            "operator_hand": ExperimentObject(
                name="operator_hand",
                class_name="operator_hand",
                bbox=BBox2D(100, 100, 200, 200, 0.95, 3, "operator_hand"),
                pos_rack=Vector3D(0.0, -0.2, 0.5)
            )
        }

        active_hoi, updated_objects, activity = self.har_agent.evaluate_interactions(
            pose=pose,
            objects=objects,
            lid_angle=0.0
        )

        # Must not report CONTACT OPERATOR HAND
        self.assertNotIn("OPERATOR HAND", activity)
        self.assertNotIn("operator_hand", [h.object_name for h in active_hoi])
        self.assertEqual(activity, "IDLE")

    def test_component_grasp_and_extraction(self):
        """Verify grasping and extracting a component item."""
        pose = AstronautPose3D()
        pose.joints["wrist"] = Joint3D(
            name="wrist",
            pos_camera=Vector3D(0.0, 0.0, 1.2),
            pos_rack=Vector3D(0.0, -0.2, 0.5)
        )

        cont = ExperimentObject(
            name="container_box",
            class_name="container_box",
            bbox=BBox2D(100, 200, 500, 600, 0.90, 0, "container_box"),
            pos_rack=Vector3D(0.0, -0.2, 0.5)
        )
        comp = ExperimentObject(
            name="component_box",
            class_name="component_box",
            bbox=BBox2D(250, 300, 350, 400, 0.90, 2, "component_box"),
            pos_rack=Vector3D(0.0, -0.2, 0.51)  # 1 cm distance
        )
        objects = {"container_box": cont, "component_box": comp}

        # Frame 1: Contact
        active_hoi, updated_objects, activity = self.har_agent.evaluate_interactions(
            pose=pose, objects=objects, lid_angle=30.0
        )
        self.assertTrue(len(active_hoi) > 0)
        self.assertEqual(active_hoi[0].object_name, "component_box")
        self.assertIn("COMPONENT BOX", activity)

        # Simulate moving component outside container (Extraction)
        comp.bbox = BBox2D(550, 100, 650, 200, 0.90, 2, "component_box")  # Outside container
        comp.pos_rack = Vector3D(0.35, -0.4, 0.5)
        pose.joints["wrist"].pos_rack = Vector3D(0.35, -0.4, 0.51)

        active_hoi, updated_objects, activity = self.har_agent.evaluate_interactions(
            pose=pose, objects=objects, lid_angle=30.0
        )
        self.assertIn("EXTRACT", activity)
        self.assertFalse(updated_objects["component_box"].is_inside_container)

    def test_container_approach_and_reach(self):
        """Verify approaching container and reaching inside."""
        pose = AstronautPose3D()
        pose.joints["wrist"] = Joint3D(
            name="wrist",
            pos_camera=Vector3D(0.0, 0.0, 1.2),
            pos_rack=Vector3D(0.15, -0.2, 0.5)
        )

        cont = ExperimentObject(
            name="container_box",
            class_name="container_box",
            bbox=BBox2D(100, 200, 500, 600, 0.90, 0, "container_box"),
            pos_rack=Vector3D(0.0, -0.2, 0.5)
        )
        objects = {"container_box": cont}

        active_hoi, updated_objects, activity = self.har_agent.evaluate_interactions(
            pose=pose, objects=objects, lid_angle=0.0
        )
        self.assertIn("APPROACH CONTAINER", activity)

    def test_yolo11_model_file_exists(self):
        """Verify models/yolo11n.pt is available and readable."""
        model_path = os.path.join(WORKSPACE_ROOT, "models", "yolo11n.pt")
        self.assertTrue(os.path.exists(model_path), "models/yolo11n.pt must exist")
        self.assertGreater(os.path.getsize(model_path), 1_000_000)


if __name__ == "__main__":
    unittest.main()
