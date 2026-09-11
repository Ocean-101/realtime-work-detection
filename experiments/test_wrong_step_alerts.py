"""
Verify wrong-step anomaly detection & alerts:
1. ERROR_SKIP: Premature close before object extraction
2. ERROR_SEQ: Premature close before object return
"""
import sys, os
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from src.core.types import BBox2D, ExperimentObject, EntityState, FSMStep, AnomalyType
from src.agents.validation_agent import ValidationAgent

validation = ValidationAgent(config_path="configs/box_return_fsm.json")

print("=" * 70)
print("TEST CASE 1: Box closed before extracting object (ERROR_SKIP)")
print("=" * 70)
validation.reset()

# 1. Open container
mock_objects = {
    "container_box": ExperimentObject(name="container_box", class_name="container_box", bbox=BBox2D(100, 200, 300, 400, 0.95, 0, "container_box")),
    "container_lid": ExperimentObject(name="container_lid", class_name="container_lid", bbox=BBox2D(100, 150, 300, 200, 0.90, 1, "container_lid")),
    "component_box": ExperimentObject(name="component_box", class_name="component_box", is_inside_container=True, state=EntityState.DOCKED)
}

# Accumulate debounce for BOX_OPENED
for f in range(validation.debounce_required):
    step, deb, anom, msg, evt = validation.evaluate_step(mock_objects, lid_angle=45.0, active_hoi=[], current_frame=f)

assert step == FSMStep.BOX_OPENED, f"Expected BOX_OPENED, got {step}"
print(f"Current step: {step.name}")

# Now prematurely close the lid (lid_angle = 5.0) for 5 frames without extracting object
wrong_step_detected = False
for f in range(10):
    step, deb, anom, msg, evt = validation.evaluate_step(mock_objects, lid_angle=5.0, active_hoi=[], current_frame=f + 10)
    if anom == AnomalyType.ERROR_SKIP:
        wrong_step_detected = True
        print(f"[ALERT TRIGGERED at frame {f+10}]: Anomaly={anom.value} | Message='{msg}' | Verdict='{validation.step_verdict}' | Correct={validation.is_step_correct}")
        break

assert wrong_step_detected, "Failed to detect ERROR_SKIP"
print(">>> TEST CASE 1 PASSED: ERROR_SKIP correctly detected and flagged!")

print("\n" + "=" * 70)
print("TEST CASE 2: Box closed before returning object (ERROR_SEQ)")
print("=" * 70)
validation.reset()

# 1. Open container
for f in range(validation.debounce_required):
    step, deb, anom, msg, evt = validation.evaluate_step(mock_objects, lid_angle=45.0, active_hoi=[], current_frame=f)

# 2. Extract object
mock_objects_extracted = {
    "container_box": mock_objects["container_box"],
    "container_lid": mock_objects["container_lid"],
    "component_box": ExperimentObject(name="component_box", class_name="component_box", is_inside_container=False, state=EntityState.EXTRACTED)
}
for f in range(validation.debounce_required):
    step, deb, anom, msg, evt = validation.evaluate_step(mock_objects_extracted, lid_angle=45.0, active_hoi=[], current_frame=f + 10)

assert step == FSMStep.OBJECT_EXTRACTED, f"Expected OBJECT_EXTRACTED, got {step}"
print(f"Current step: {step.name}")

# 3. Prematurely close lid without returning object
wrong_step_seq_detected = False
for f in range(10):
    step, deb, anom, msg, evt = validation.evaluate_step(mock_objects_extracted, lid_angle=5.0, active_hoi=[], current_frame=f + 20)
    if anom == AnomalyType.ERROR_SEQ:
        wrong_step_seq_detected = True
        print(f"[ALERT TRIGGERED at frame {f+20}]: Anomaly={anom.value} | Message='{msg}' | Verdict='{validation.step_verdict}' | Correct={validation.is_step_correct}")
        break

assert wrong_step_seq_detected, "Failed to detect ERROR_SEQ"
print(">>> TEST CASE 2 PASSED: ERROR_SEQ correctly detected and flagged!")
print("\n" + "=" * 70)
print("ALL WRONG-STEP ANOMALY TESTS PASSED!")
print("=" * 70)
