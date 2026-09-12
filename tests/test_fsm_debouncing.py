"""
Unit Tests for Deterministic Finite State Machine (FSM) & 15-Frame Debouncing
ISRO Smart India Hackathon (SIH) | Problem Statement ID: 26174
Validates step transitions, temporal debouncing, and out-of-order anomaly alarms.
"""

import os
import sys
import unittest

# Ensure workspace root in path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import FSMStep, AnomalyType, EntityState, ExperimentObject, BBox2D
from src.agents.validation_agent import ValidationAgent


class TestFSMDebouncing(unittest.TestCase):

    def setUp(self):
        self.validator = ValidationAgent(config_path="configs/box_return_fsm.json")
        self.deb_req = self.validator.debounce_required

    def test_nominal_step_progression_with_debounce(self):
        """Verify that state only advances after exactly debounce_required consecutive frames."""
        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
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

        # First 7 frames accumulate anomaly debounce
        for f in range(7):
            step, deb, anomaly, msg, trans = self.validator.evaluate_step(objects, lid_angle=5.0, active_hoi=[], current_frame=f)
            self.assertEqual(anomaly, AnomalyType.NONE)

        # 8th consecutive frame trips ERROR_SKIP
        step, deb, anomaly, msg, trans = self.validator.evaluate_step(objects, lid_angle=5.0, active_hoi=[], current_frame=8)
        self.assertEqual(step, FSMStep.BOX_OPENED)
        self.assertEqual(anomaly, AnomalyType.ERROR_SKIP)
        self.assertIn("Warning: Step skipped", msg)

    def test_box_return_fsm_nominal_flow(self):
        """Validates nominal progression for the active Box Object Extraction & Return procedure."""
        val = ValidationAgent(config_path="configs/box_return_fsm.json")
        deb = val.debounce_required
        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
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
        self.assertTrue(val.is_step_correct)
        self.assertIn("NOMINAL", val.step_verdict)

    def test_isro_dual_box_nominal_flow(self):
        """Validates ISRO Benchmark PS #26174 nominal flow (Red then Yellow)."""
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-26174")
        deb = val.debounce_required

        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True),
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Step 0 -> Step 1: Open Container Box
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=35.0, active_hoi=[], current_frame=f)
        self.assertEqual(step, FSMStep.CONTAINER_OPEN)
        self.assertEqual(trans, "CONTAINER_OPENED")
        self.assertTrue(val.is_step_correct)

        # Step 1 -> Step 2: Extract Red Box
        objects["red_box"].is_inside_container = False
        objects["red_box"].state = EntityState.EXTRACTED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=35.0, active_hoi=[], current_frame=deb + f)
        self.assertEqual(step, FSMStep.RED_EXTRACTED)
        self.assertEqual(trans, "RED_BOX_EXTRACTED")
        self.assertTrue(val.is_step_correct)

        # Step 2 -> Step 3: Extract Yellow Box
        objects["yellow_box"].is_inside_container = False
        objects["yellow_box"].state = EntityState.EXTRACTED
        for f in range(deb):
            step, _, _, _, trans = val.evaluate_step(objects, lid_angle=35.0, active_hoi=[], current_frame=2 * deb + f)
        self.assertEqual(step, FSMStep.YELLOW_EXTRACTED)
        self.assertEqual(trans, "YELLOW_BOX_EXTRACTED")
        self.assertTrue(val.is_step_correct)

    def test_isro_dual_box_sequence_anomaly(self):
        """Validates that extracting Yellow Box before Red Box triggers ERROR_SEQ and marks step incorrect."""
        val = ValidationAgent(config_path="configs/experiment_fsm.json")
        deb = val.debounce_required

        objects = {
            "container_box": ExperimentObject(name="container_box", class_name="container_box", state=EntityState.DOCKED),
            "red_box": ExperimentObject(name="red_box", class_name="red_box", state=EntityState.DOCKED, is_inside_container=True),
            "yellow_box": ExperimentObject(name="yellow_box", class_name="yellow_box", state=EntityState.DOCKED, is_inside_container=True)
        }

        # Step 0 -> Step 1: Open Container
        for f in range(deb):
            val.evaluate_step(objects, lid_angle=40.0, active_hoi=[], current_frame=f)
        self.assertEqual(val.current_step, FSMStep.CONTAINER_OPEN)

        # Astronaut attempts to extract Yellow Box before Red Box!
        objects["yellow_box"].is_inside_container = False
        objects["yellow_box"].state = EntityState.EXTRACTED

        for f in range(5):
            step, _, anomaly, msg, _ = val.evaluate_step(objects, lid_angle=40.0, active_hoi=[], current_frame=deb + f)

        self.assertEqual(anomaly, AnomalyType.ERROR_SEQ)
        self.assertFalse(val.is_step_correct)
        self.assertIn("PROCEDURAL ERROR", val.step_verdict)
        self.assertIn("yellow box", msg.lower())

    def test_dynamic_protocol_switching(self):
        """Verifies ValidationAgent dynamically switches protocol between Box Return and ISRO Dual-Box."""
        val = ValidationAgent(config_path="configs/box_return_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-BOX-RETURN")

        val.load_protocol("configs/experiment_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-26174")
        self.assertEqual(val.current_step, FSMStep.IDLE)
        self.assertTrue(val.is_step_correct)

        val.load_protocol("configs/box_return_fsm.json")
        self.assertEqual(val.experiment_id, "BAS-EXP-BOX-RETURN")
        self.assertEqual(val.current_step, FSMStep.IDLE)


if __name__ == "__main__":
    unittest.main()
