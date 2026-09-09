"""
Unit Tests for Deterministic Validation Agent (FSM) & Debounce Logic
"""

import unittest
from src.core.types import (
    FSMStep,
    AnomalyType,
    ExperimentObject,
    EntityState,
    Vector3D
)
from src.agents.validation_agent import ValidationAgent


class TestFSMValidation(unittest.TestCase):

    def setUp(self):
        self.validator = ValidationAgent()

    def test_nominal_step_progression_with_debounce(self):
        """Verify that state only advances after exactly 15 consecutive frames."""
        objects = {
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True),
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Step 0 -> Step 1: Open Lid
        # Provide 14 frames of lid open evidence
        for f in range(14):
            step, deb, anomaly, _, trans = self.validator.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=f)
            self.assertEqual(step, FSMStep.IDLE, f"Step should remain IDLE on frame {f}")
            self.assertEqual(deb, f + 1)
            self.assertIsNone(trans)

        # 15th frame commits the transition
        step, deb, anomaly, _, trans = self.validator.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=15)
        self.assertEqual(step, FSMStep.CONTAINER_OPEN)
        self.assertEqual(trans, "CONTAINER_OPENED")
        self.assertEqual(anomaly, AnomalyType.NONE)

    # =========================================================================
    # TEMPORARILY COMMENTED OUT: Red and Yellow Box Extraction & Anomaly Tests
    # =========================================================================
    # def test_out_of_sequence_anomaly(self):
    #     """Verify that touching/extracting yellow box in State 1 flags ERROR_SEQ."""
    #     # Force validator into State 1 (CONTAINER_OPEN)
    #     self.validator.current_step = FSMStep.CONTAINER_OPEN
    #
    #     # Astronaut manipulates Yellow Box before Red Box
    #     objects = {
    #         "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True),
    #         "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.GRASPED, is_inside_container=False)
    #     }
    #
    #     step, deb, anomaly, msg, trans = self.validator.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=1)
    #     self.assertEqual(step, FSMStep.CONTAINER_OPEN, "FSM should refuse transition on anomaly")
    #     self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
    #     self.assertIn("Warning: Procedural error", msg)
    #
    # def test_sequence_skip_anomaly(self):
    #     """Verify that closing lid before extracting yellow box in State 2 flags ERROR_SKIP."""
    #     # Force validator into State 2 (RED_EXTRACTED)
    #     self.validator.current_step = FSMStep.RED_EXTRACTED
    #
    #     # Astronaut closes lid while yellow box is still inside
    #     objects = {
    #         "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.EXTRACTED, is_inside_container=False),
    #         "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.DOCKED, is_inside_container=True)
    #     }
    #
    #     step, deb, anomaly, msg, trans = self.validator.evaluate_step(objects, lid_angle=5.0, active_hoi=[], current_frame=1)
    #     self.assertEqual(step, FSMStep.RED_EXTRACTED)
    #     self.assertEqual(anomaly, AnomalyType.ERROR_SKIP)
    #     self.assertIn("Warning: Step skipped", msg)

    def test_box_return_fsm_nominal_flow(self):
        """Validates nominal progression for the active Box Object Extraction & Return procedure."""
        val = ValidationAgent(config_path="configs/box_return_fsm.json")
        objects = {
            "component_box": ExperimentObject(name="component_box", class_name="box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # S0 -> S1: Box Opening (12 frames)
        for f in range(12):
            step, deb, anomaly, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=f)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(trans, "BOX_OPENED")

        # S1 -> S2: Object Extracted (12 frames)
        objects["component_box"].is_inside_container = False
        objects["component_box"].state = EntityState.EXTRACTED
        for f in range(12):
            step, deb, anomaly, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=12 + f)
        self.assertEqual(step, FSMStep.OBJECT_EXTRACTED)
        self.assertEqual(trans, "OBJECT_EXTRACTED")


if __name__ == "__main__":
    unittest.main()
