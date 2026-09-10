"""
Unit Tests for Deterministic Validation Agent (FSM) & Debounce Logic
"""

import unittest
from src.core.types import (
    FSMStep,
    AnomalyType,
    ExperimentObject,
    EntityState
)
from src.agents.validation_agent import ValidationAgent


class TestFSMValidation(unittest.TestCase):

    def setUp(self):
        self.validator = ValidationAgent()
        self.deb_req = self.validator.debounce_required

    def test_nominal_step_progression_with_debounce(self):
        """Verify that state only advances after exactly debounce_required consecutive frames."""
        objects = {
            "component_box": ExperimentObject(name="component_box", class_name="component_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Step 0 -> Step 1: Open Lid (deb_req frames)
        for f in range(self.deb_req - 1):
            step, deb, anomaly, _, trans = self.validator.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=f)
            self.assertEqual(step, FSMStep.IDLE, f"Step should remain IDLE on frame {f}")
            self.assertEqual(deb, f + 1)
            self.assertIsNone(trans)

        # Final frame commits the transition
        step, deb, anomaly, _, trans = self.validator.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=self.deb_req - 1)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(trans, "BOX_OPENED")
        self.assertEqual(anomaly, AnomalyType.NONE)

    def test_premature_close_anomaly(self):
        """Verify that closing lid in Step 1 before extracting object flags ERROR_SKIP after debouncing."""
        self.validator.current_step = FSMStep.BOX_OPENED
        objects = {
            "component_box": ExperimentObject(name="component_box", class_name="component_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # First 4 frames accumulate anomaly debounce
        for f in range(4):
            step, deb, anomaly, msg, trans = self.validator.evaluate_step(objects, lid_angle=5.0, active_hoi=[], current_frame=f)
            self.assertEqual(anomaly, AnomalyType.NONE)

        # 5th consecutive frame trips ERROR_SKIP
        step, deb, anomaly, msg, trans = self.validator.evaluate_step(objects, lid_angle=5.0, active_hoi=[], current_frame=4)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(anomaly, AnomalyType.ERROR_SKIP)
        self.assertIn("Warning: Step skipped", msg)

    def test_box_return_fsm_nominal_flow(self):
        """Validates nominal progression for the active Box Object Extraction & Return procedure."""
        val = ValidationAgent(config_path="configs/box_return_fsm.json")
        deb = val.debounce_required
        objects = {
            "component_box": ExperimentObject(name="component_box", class_name="component_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # S0 -> S1: Box Opening
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=f)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(trans, "BOX_OPENED")

        # S1 -> S2: Object Extracted
        objects["component_box"].is_inside_container = False
        objects["component_box"].state = EntityState.EXTRACTED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=deb + f)
        self.assertEqual(step, FSMStep.OBJECT_EXTRACTED)
        self.assertEqual(trans, "OBJECT_EXTRACTED")

        # S2 -> S3: Object Returned
        objects["component_box"].is_inside_container = True
        objects["component_box"].state = EntityState.DOCKED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=60.0, active_hoi=[], current_frame=2 * deb + f)
        self.assertEqual(step, FSMStep.OBJECT_RETURNED)
        self.assertEqual(trans, "OBJECT_RETURNED")

        # S3 -> S4: Box Closed
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=10.0, active_hoi=[], current_frame=3 * deb + f)
        self.assertEqual(step, FSMStep.COMPLETE)
        self.assertEqual(trans, "BOX_CLOSED")


if __name__ == "__main__":
    unittest.main()
